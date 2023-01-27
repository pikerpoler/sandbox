import os
import abc
import sys
import tqdm
import torch
from typing import Any, Callable
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from torch import optim

from typing import List, NamedTuple


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    att_loss: float
    class_loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """

    losses: List[float]
    att_losses: List[float]
    class_losses: List[float]
    accuracy: float



class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device="cpu", reportToWandb=False):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)
        self.reportToWandb = reportToWandb
        self.scheduler = None

    def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every=1,
        post_epoch_fn=None,
        reportToWandb=False,
        **kw,
    ) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :param reportToWandb: Whether to report to wandb.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        epochs_without_improvement = 0

        # init scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_epochs)

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f"{checkpoints}.pt"
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f"*** Loading checkpoint file {checkpoint_filename}")
                saved_state = torch.load(checkpoint_filename, map_location=self.device)
                best_acc = saved_state.get("best_acc", best_acc)
                epochs_without_improvement = saved_state.get(
                    "ewi", epochs_without_improvement
                )
                self.model.load_state_dict(saved_state["model_state"])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f"--- EPOCH {epoch+1}/{num_epochs} ---", verbose)
            wandb.watch(self.model)
            self.optimizer.zero_grad()
            train_result = self.train_epoch(dl_train, verbose=verbose, reportToWandb=reportToWandb, **kw)
            test_result = self.test_epoch(dl_test, verbose=verbose, reportToWandb=reportToWandb, **kw)
            train_loss.extend(train_result.losses)
            train_acc.append(train_result.accuracy)
            test_loss.extend(test_result.losses)
            test_acc.append(test_result.accuracy)
            acc_len = len(test_acc)
            if not(checkpoints is None) and acc_len > 1 and test_acc[-1] > test_acc[-2]:
                save_checkpoint = True
            no_improve = True
            if early_stopping and acc_len > early_stopping:
                curr_acc = test_acc[-1]
                for index in range(1, early_stopping+1):
                    if curr_acc > test_acc[acc_len - index]:
                        no_improve = False
            else:
                no_improve = False
            if early_stopping and no_improve:
                return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(
                    best_acc=best_acc,
                    ewi=epochs_without_improvement,
                    model_state=self.model.state_dict(),
                )
                torch.save(saved_state, checkpoint_filename)
                print(
                    f"*** Saved checkpoint {checkpoint_filename} " f"at epoch {epoch+1}"
                )

            if post_epoch_fn:
                post_epoch_fn(epoch, train_result, test_result, verbose)

            # update scheduler
            self.scheduler.step()
            # print("lr: ", self.scheduler.get_lr())
        return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, reportToWandb, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :param reportToWandb: Whether to report to wandb.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        lr = float(self.scheduler.get_last_lr()[0])
        return self._foreach_batch(dl_train, self.train_batch, reportToWandb=reportToWandb, train=True, lr=lr, **kw)

    def test_epoch(self, dl_test: DataLoader, reportToWandb, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :param reportToWandb: If true, report to wandb
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, train=False, reportToWandb=reportToWandb, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(
            dl: DataLoader,
            forward_fn: Callable[[Any], BatchResult],
            verbose: object = True,
            max_batches: object = None,
            reportToWandb: object = False,
            train: object = True,
            lr: object = None,
    ) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        att_losses = []
        class_losses = []

        bb_hc_weight_ratios = []
        variances = []
        means = []
        medians = []
        ginis = []
        AUROCs = []
        AURCs = []

        att_accuracies = []
        att_f1s = []
        att_precisions = []
        att_recalls = []

        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, "w")

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches, file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f"{pbar_name} ({batch_res.loss:.3f})")
                pbar.update()

                losses.append(batch_res.loss)
                att_losses.append(batch_res.att_loss)
                class_losses.append(batch_res.class_loss)
                num_correct += batch_res.num_correct

                bb_hc_weight_ratios.append(batch_res.bb_hc_weight_ratio)
                variances.append(batch_res.variance)
                means.append(batch_res.mean)
                medians.append(batch_res.median)
                ginis.append(batch_res.gini)
                AUROCs.append(batch_res.AUROC)
                AURCs.append(batch_res.AURC)

                att_accuracies.append(batch_res.att_accuracy)
                att_f1s.append(batch_res.att_f1)
                att_precisions.append(batch_res.att_precision)
                att_recalls.append(batch_res.att_recall)

            avg_loss = sum(losses) / num_batches
            avg_att_loss = sum(att_losses) / num_batches
            avg_class_loss = sum(class_losses) / num_batches
            accuracy = num_correct / num_samples

            # att_f1, att_precision, att_recall, att_accuracy = multilabel_f1(atts, gt_atts)

            avg_bb_hc_weight_ratio = sum(bb_hc_weight_ratios) / num_batches
            avg_variance = sum(variances) / num_batches
            avg_mean = sum(means) / num_batches
            avg_median = sum(medians) / num_batches
            avg_gini = sum(ginis) / num_batches
            avg_AUROC = sum(AUROCs) / num_batches
            avg_AURC = sum(AURCs) / num_batches

            avg_att_accuracy = sum(att_accuracies) / num_batches
            avg_att_f1 = sum(att_f1s) / num_batches
            avg_att_precision = sum(att_precisions) / num_batches
            avg_att_recall = sum(att_recalls) / num_batches

            pbar.set_description(
                f"{pbar_name} "
                f"(Avg. Loss {avg_loss:.3f}, "
                f"Avg. Att. Loss {avg_att_loss:.3f}, "
                f"Avg. Class Loss {avg_class_loss:.3f}, "
                f"Accuracy {accuracy:.3f})"
            )
            # report to wandb
            if reportToWandb:
                print("reporting to wandb")
                wandb.log(
                    {
                        # "{} loss".format('train' if train else 'test'): avg_loss,
                        # "{} attribute loss".format('train' if train else 'test'): avg_att_loss,
                        # "{} class loss".format('train' if train else 'test'): avg_class_loss,
                        # "{} accuracy".format('train' if train else 'test'): accuracy,
                        "{}/Att accuracy".format('train' if train else 'test'): avg_att_accuracy,
                        "{}/Att f1".format('train' if train else 'test'): avg_att_f1,
                        "{}/Att precision".format('train' if train else 'test'): avg_att_precision,
                        "{}/Att recall".format('train' if train else 'test'): avg_att_recall,
                        "{}/bb_hc_weight_ratio".format('train' if train else 'test'): avg_bb_hc_weight_ratio,
                        "Uncertainty_{}/variance".format('train' if train else 'test'): avg_variance,
                        "Uncertainty_{}/mean".format('train' if train else 'test'): avg_mean,
                        "Uncertainty_{}/median".format('train' if train else 'test'): avg_median,
                        "Uncertainty_{}/gini".format('train' if train else 'test'): avg_gini,
                        "Uncertainty_{}/AUROC".format('train' if train else 'test'): avg_AUROC,
                        "Uncertainty_{}/AURC".format('train' if train else 'test'): avg_AURC,
                        "{}/learning rate".format('train' if train else 'test'): 0 if lr is None else lr,
                        "{}/loss".format('train' if train else 'test'): avg_loss,
                        "{}/attribute loss".format('train' if train else 'test'): avg_att_loss,
                        "{}/class loss".format('train' if train else 'test'): avg_class_loss,
                        "{}/accuracy".format('train' if train else 'test'): accuracy,
                        "{}/learning rate".format('train' if train else 'test'): 0 if lr is None else lr,
                    }
                )
            else:
                print("not reporting to wandb")

        return EpochResult(att_losses=att_losses, class_losses=class_losses, losses=losses, accuracy=accuracy
                           , att_accuracies=att_accuracies, att_f1s=att_f1s, att_precisions=att_precisions,
                           att_recalls=att_recalls)



class GeneralResnetTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None, reportToWandb=False):
        super().__init__(model, loss_fn, optimizer, device, reportToWandb)

    def train_batch(self, batch) -> BatchResult:
        imgs, labels = batch
        self.optimizer.zero_grad()
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        output = self.model(imgs).to(self.device)

        # make predictions zero/one
        loss = self.loss_fn(output, labels)
        pred = round(output)
        num_correct = torch.eq(pred, labels).sum().item()
        num_correct = num_correct / labels.shape[-1]
        loss.backward()
        self.optimizer.step()
        return BatchResult(loss.item(), loss.item(), loss.item(), num_correct)

    def test_batch(self, batch) -> BatchResult:
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)

        with torch.no_grad():
            output = self.model(imgs)
            loss = self.loss_fn(output, labels)
            num_correct = torch.eq(output, labels).sum().item()

        return BatchResult(loss.item(), loss.item(), loss.item(), num_correct)
