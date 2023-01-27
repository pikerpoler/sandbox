import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, ResNet18_Weights, ResNet34_Weights, \
    ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class ResNetCLF(torch.nn.Module):
    def __init__(self, num_classes=10, depth=18, pretrained=False, in_channels=3, dropout=0.0):
        super(ResNetCLF, self).__init__()
        if depth == 18:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif depth == 34:
            self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif depth == 50:
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif depth == 101:
            self.resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        elif depth == 152:
            self.resnet = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.training:
            x = self.dropout(x)
        x = self.resnet(x)
        return x

class TwoHeadedResNetCLF(torch.nn.Module):
    def __init__(self, num_classes=(4, 6), depth=18, pretrained=False):
        super(TwoHeadedResNetCLF, self).__init__()
        if depth == 18:
            self.resnet = resnet18()
        elif depth == 34:
            self.resnet = resnet34()
        elif depth == 50:
            self.resnet = resnet50()
        elif depth == 101:
            self.resnet = resnet101()
        elif depth == 152:
            self.resnet = resnet152()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features, num_classes[0])
        self.fc2 = nn.Linear(num_features, num_classes[1])


    def forward(self, x):
        x = self.resnet(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2

class CompoundCLF(torch.nn.Module):
    def __init__(self, groupclf, headclf):
        super(CompoundCLF, self).__init__()
        self.groupclf = groupclf
        self.headclf = headclf

    def forward(self, x):
        group_scores = self.groupclf(x)
        x1, x2 = self.headclf(x)
        x = x1 * group_scores[:, 0].unsqueeze(1) + x2 * group_scores[:, 1].unsqueeze(1)
        return x



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CifarDataset(Dataset):
    def __init__(self, path='/datasets/cifar-10-batches-py/data_batch_1', transform=None, add_noise=False):
        self.path = path
        self.transform = transform
        self.add_noise = add_noise
        self.data, self.labels = self._make_dataset()

    def _make_dataset(self):
        cifar_dict = unpickle(self.path)
        data = cifar_dict[b'data']
        labels = cifar_dict[b'labels']
        data = data.reshape((10000, 3, 32, 32))
        data = data / 255.0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
        if self.add_noise:
            noise = torch.rand(data.shape[0], 1, data.shape[2], data.shape[3])
            data = torch.cat((data, noise), dim=1)
        labels = torch.from_numpy(np.array(labels))
        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def check_dataset(dataset):
    dl = DataLoader(dataset, batch_size=1, shuffle=True)
    labels = {}
    for i, (x, y) in enumerate(dl):
        if y.item() not in labels:
            labels[y.item()] = 0
        labels[y.item()] += 1
    print(labels)

def n_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(batch_loop, dataloader, val_dataloader, scheduler, name='dummy', num_epochs=50, save_plot=False):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    print(f'starting training {name}')
    for epoch in range(num_epochs):
        eopoch_losses = []
        eopoch_accuracies = []
        for i, (images, labels) in enumerate(dataloader):
            loss, accuracy = batch_loop(images, labels)
            eopoch_losses.append(loss)
            eopoch_accuracies.append(accuracy)
        train_losses.append(sum(eopoch_losses) / len(eopoch_losses))
        train_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
        eopoch_losses = []
        eopoch_accuracies = []
        scheduler.step()
        for i, (images, labels) in enumerate(val_dataloader):
            with torch.no_grad():
                loss, accuracy = batch_loop(images, labels, eval=True)
                eopoch_losses.append(loss)
                eopoch_accuracies.append(accuracy)
        val_losses.append(sum(eopoch_losses) / len(eopoch_losses))
        val_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
        if save_plot:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].plot(train_losses, label='train loss')
            ax[0, 0].set_title('train loss')
            ax[0, 1].plot(val_losses, label='val loss')
            ax[0, 1].set_title('val loss')
            ax[1, 0].plot(train_accuracies, label='train accuracy')
            ax[1, 0].set_title('train accuracy')
            ax[1, 1].plot(val_accuracies, label='val accuracy')
            ax[1, 1].set_title('val accuracy')
            ax[0, 0].legend()
            # if plots folder doesn't exist, create it
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.savefig(f'plots/resnet_experiment_{name}_{epoch}.png')
            plt.close()
    return train_losses, train_accuracies, val_losses, val_accuracies

def resnet_experiment_0():
    """
    this experiment is meant to find a good set of hyperparameters to serve as a baseline for further experiments
    """
    # hyperparameters
    num_epochs = 50
    batch_size = 400
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    dataset = CifarDataset(path='/datasets/cifar-10-batches-py/data_batch_1')
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
    val_dataset = CifarDataset(path='/datasets/cifar-10-batches-py/test_batch')
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=2)

    learning_rates = [0.01, 0.001]
    weight_decays = [0.001]
    dropouts = [0.0, 0.2, 0.5]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].set_title('train loss')
    ax[0, 1].set_title('val loss')
    ax[1, 0].set_title('train accuracy')
    ax[1, 1].set_title('val accuracy')

    for dropout in dropouts:
        for lr in learning_rates:
            for wd in weight_decays:
                model = ResNetCLF(dropout=dropout).to(device)
                name = f'lr_{lr}_wd_{wd}_dropout_{dropout}'
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                loss = nn.CrossEntropyLoss()

                def batch_loop(images, labels, eval=False):
                    images = images.to(device)
                    labels = labels.to(device)
                    if not eval:
                        model.train()
                        optimizer.zero_grad()
                    else:
                        model.eval()
                    outputs = model(images)
                    loss_value = loss(outputs, labels)
                    if not eval:
                        loss_value.backward()
                        optimizer.step()
                    accuracy = (outputs.argmax(dim=1) == labels).float().mean()
                    return loss_value.item(), accuracy.item()

                train_losses, train_accuracies, val_losses, val_accuracies = train(batch_loop, dataloader, val_dataloader, scheduler, name, num_epochs=num_epochs, save_plot=False)
                ax[0, 0].plot(train_losses, label=name)
                ax[0, 1].plot(val_losses, label=name)
                ax[1, 0].plot(train_accuracies, label=name)
                ax[1, 1].plot(val_accuracies, label=name)

    ax[0, 0].legend()
    if not os.path.exists('plots'):
        os.makedirs('plots')
    fig.savefig(f'plots/resnet_experiment_0.png')
    plt.close()


def resnet_experiment_1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    models = {
        "model_1": ResNetCLF(depth=18).to(device),
        "model_2": ResNetCLF(depth=34).to(device),
    }

    for model_name, model in models.items():
        print(f'Number of parameters for {model_name}: {n_parameters(model)}')

    loss_fun = nn.CrossEntropyLoss()
    learning_rates = [0.0001, 0.00001]
    optimizers = {(model_name, lr): torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) for model_name, model in models.items() for lr in learning_rates}
    schedulers = {(model_name, lr): torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs) for (model_name, lr), optimizer in optimizers.items()}
    train_losses = {(model_name, lr): [] for model_name, model in models.items() for lr in learning_rates}
    val_losses = {(model_name, lr): [] for model_name, model in models.items() for lr in learning_rates}
    train_accuracies = {(model_name, lr): [] for model_name, model in models.items() for lr in learning_rates}
    val_accuracies = {(model_name, lr): [] for model_name, model in models.items() for lr in learning_rates}

    def batch_loop(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device).to(torch.long)
        outputs = model(images)
        loss = loss_fun(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), accuracy

    dataset = CifarDataset(path='/datasets/cifar-10-batches-py/data_batch_1')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = CifarDataset(path='/datasets/cifar-10-batches-py/test_batch')
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    check_dataset(dataset)
    check_dataset(val_dataset)

    for epoch in range(1, num_epochs):
        print(f'epoch: {epoch}')
        for model_name, model in models.items():
            for lr in learning_rates:
                optimizer = optimizers[(model_name, lr)]
                scheduler = schedulers[(model_name, lr)]
                eopoch_losses = []
                eopoch_accuracies = []
                for i, (images, labels) in enumerate(dataloader):
                    loss, accuracy = batch_loop(images, labels)
                    eopoch_losses.append(loss)
                    eopoch_accuracies.append(accuracy)
                train_losses[(model_name, lr)].append(sum(eopoch_losses) / len(eopoch_losses))
                train_accuracies[(model_name, lr)].append(sum(eopoch_accuracies) / len(eopoch_accuracies))
                eopoch_losses = []
                eopoch_accuracies = []
                scheduler.step()
                for i, (images, labels) in enumerate(val_dataloader):
                    with torch.no_grad():
                        loss, accuracy = batch_loop(images, labels, eval=True)
                        eopoch_losses.append(loss)
                        eopoch_accuracies.append(accuracy)
                val_losses[(model_name, lr)].append(sum(eopoch_losses) / len(eopoch_losses))
                val_accuracies[(model_name, lr)].append(sum(eopoch_accuracies) / len(eopoch_accuracies))

        # plot losses for all models, one for train and one for val, side by side
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2)
        for model_name, model in models.items():
            for lr in learning_rates:
                ax[0, 0].plot(train_losses[(model_name, lr)], label=f'{model_name} lr: {lr}')
                ax[0, 0].set_title(f'train loss')
                ax[0, 1].plot(val_losses[(model_name, lr)], label=f'{model_name} lr: {lr}')
                ax[0, 1].set_title('val loss')
                ax[1, 0].plot(train_accuracies[(model_name, lr)], label=f'{model_name} lr: {lr}')
                ax[1, 0].set_title(f'train accuracy')
                ax[1, 1].plot(val_accuracies[(model_name, lr)], label=f'{model_name} lr: {lr}')
                ax[1, 1].set_title(f'val accuracy')


        ax[0, 0].legend()
        # if plots folder doesn't exist, create it
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.savefig(f'plots/resnet_experiment_1_{epoch}.png')
        plt.close()
        loss_dict = {"val": val_losses, "train": train_losses}
        torch.save(loss_dict, 'plots/resnet_experiment_1_losses.pt')

def resnet_experiment_2():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    # cifar10 dataset classes are  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # we will divide them into two groups: ['airplane', 'automobile', 'ship', 'truck'] and ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    # first we will train a model to classify between the groups, then we will train a model to classify between the classes

    group1 = ['airplane', 'automobile', 'ship', 'truck']
    group2 = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    group1_indices = [class_names.index(name) for name in group1]
    group2_indices = [class_names.index(name) for name in group2]


    dataset = CifarDataset(path='/datasets/cifar-10-batches-py/data_batch_1')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = CifarDataset(path='/datasets/cifar-10-batches-py/test_batch')
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    def train(batch_loop, scheduler):
        for epoch in range(num_epochs):
            eopoch_losses = []
            eopoch_accuracies = []
            for i, (images, labels) in enumerate(dataloader):
                loss, accuracy = batch_loop(images, labels)
                eopoch_losses.append(loss)
                eopoch_accuracies.append(accuracy)
            train_losses.append(sum(eopoch_losses) / len(eopoch_losses))
            train_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
            eopoch_losses = []
            eopoch_accuracies = []
            scheduler.step()
            for i, (images, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    loss, accuracy = batch_loop(images, labels, eval=True)
                    eopoch_losses.append(loss)
                    eopoch_accuracies.append(accuracy)
            val_losses.append(sum(eopoch_losses) / len(eopoch_losses))
            val_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
            print(f'epoch: {epoch}')
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].plot(train_losses, label='train loss')
            ax[0, 0].set_title('train loss')
            ax[0, 1].plot(val_losses, label='val loss')
            ax[0, 1].set_title('val loss')
            ax[1, 0].plot(train_accuracies, label='train accuracy')
            ax[1, 0].set_title('train accuracy')
            ax[1, 1].plot(val_accuracies, label='val accuracy')
            ax[1, 1].set_title('val accuracy')
            ax[0, 0].legend()
            # if plots folder doesn't exist, create it
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.savefig(f'plots/resnet_experiment_2_{epoch}.png')
            plt.close()




    ######################################################################
    groupclf = ResNetCLF(num_classes=2)
    groupclf.to(device)
    num_epochs = 5
    groupclf_optimizer = optim.SGD(groupclf.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    groupclf_scheduler = optim.lr_scheduler.CosineAnnealingLR(groupclf_optimizer, T_max=num_epochs)
    groupclf_loss_fun = nn.CrossEntropyLoss()

    def batch_loop_1(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device)
        for idx in group1_indices:
            labels[labels == idx] = -1
        for idx in group2_indices:
            labels[labels == idx] = -2
        labels[labels == -1] = 0
        labels[labels == -2] = 1

        outputs = groupclf(images)
        loss = groupclf_loss_fun(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        if not eval:
            groupclf_optimizer.zero_grad()
            loss.backward()
            groupclf_optimizer.step()
        return loss.item(), accuracy





    ######################################################################

    # two_headed_resnet = TwoHeadedResNetCLF(num_classes=(len(group1), len(group2)))
    two_headed_resnet = TwoHeadedResNetCLF(num_classes=(10, 10))
    two_headed_resnet.to(device)

    compoundclf = CompoundCLF(groupclf, two_headed_resnet)
    compoundclf.to(device)
    num_epochs = 5
    compoundclf_optimizer = optim.SGD(compoundclf.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    compoundclf_scheduler = optim.lr_scheduler.CosineAnnealingLR(compoundclf_optimizer, T_max=num_epochs)

    def batch_loop_2(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = compoundclf(images)
        loss = groupclf_loss_fun(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        if not eval:
            compoundclf_optimizer.zero_grad()
            loss.backward()
            compoundclf_optimizer.step()
        return loss.item(), accuracy


    ######################################################################

    num_epochs = 10
    two_headed_resnet_optimizer = optim.SGD(two_headed_resnet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    two_headed_resnet_scheduler = optim.lr_scheduler.CosineAnnealingLR(two_headed_resnet_optimizer, T_max=num_epochs)
    two_headed_resnet_loss_fun = nn.CrossEntropyLoss()

    def batch_loop_3(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device)
        x1, x2 = two_headed_resnet(images)
        group1_mask = torch.zeros_like(labels)
        group2_mask = torch.zeros_like(labels)
        for i, label in enumerate(labels):
            if label in group1_indices:
                group1_mask[i] = 1
            else:
                group2_mask[i] = 1
        x1 = x1 * group1_mask.unsqueeze(1)
        x2 = x2 * group2_mask.unsqueeze(1)
        loss1 = two_headed_resnet_loss_fun(x1, labels)
        loss2 = two_headed_resnet_loss_fun(x2, labels)
        # calculate the total loss
        loss = loss1 + loss2
        # calculate the total accuracy
        accuracy = (torch.sum(torch.argmax(x1, dim=1) == labels) + torch.sum(torch.argmax(x2, dim=1) == labels)) / len(
            labels)
        if not eval:
            two_headed_resnet_optimizer.zero_grad()
            loss.backward()
            two_headed_resnet_optimizer.step()
        return loss.item(), accuracy.cpu().detach().numpy()

    train(batch_loop_1, groupclf_scheduler)
    train(batch_loop_3, two_headed_resnet_scheduler)
    train(batch_loop_2, compoundclf_scheduler)


def resnet_experiment_3():
    """
    this experiment is trying to check
    weather concatenating noise to the input of the model will confuse the model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    dataset = CifarDataset(path='/datasets/cifar-10-batches-py/data_batch_1', add_noise=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    val_dataset = CifarDataset(path='/datasets/cifar-10-batches-py/test_batch', add_noise=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=2)

    noise_model = ResNetCLF(in_channels=4, num_classes=10,pretrained=True)
    model = ResNetCLF(in_channels=3, num_classes=10, pretrained=True)

    # train the noise model
    num_epochs = 50
    noise_model.to(device)
    model.to(device)
    noise_model_optimizer = optim.SGD(noise_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    noise_model_scheduler = optim.lr_scheduler.CosineAnnealingLR(noise_model_optimizer, T_max=num_epochs)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    noise_model_loss_fun = nn.CrossEntropyLoss()


    def batch_loop_1(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = noise_model(images)
        loss = noise_model_loss_fun(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        if not eval:
            noise_model_optimizer.zero_grad()
            loss.backward()
            noise_model_optimizer.step()
        return loss.item(), accuracy

    def batch_loop_2(images, labels, eval=False):
        images = images.to(device)
        labels = labels.to(device)
        # remove the last channel
        images = images[:, :-1, :, :].to(device)
        outputs = model(images)
        loss = noise_model_loss_fun(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(preds == labels).item() / len(labels)
        if not eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss.item(), accuracy


    def train(batch_loop, scheduler, name=''):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        print(f'starting training {name}')
        for epoch in range(num_epochs):
            eopoch_losses = []
            eopoch_accuracies = []
            for i, (images, labels) in enumerate(dataloader):
                loss, accuracy = batch_loop(images, labels)
                eopoch_losses.append(loss)
                eopoch_accuracies.append(accuracy)
            train_losses.append(sum(eopoch_losses) / len(eopoch_losses))
            train_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
            eopoch_losses = []
            eopoch_accuracies = []
            scheduler.step()
            for i, (images, labels) in enumerate(val_dataloader):
                with torch.no_grad():
                    loss, accuracy = batch_loop(images, labels, eval=True)
                    eopoch_losses.append(loss)
                    eopoch_accuracies.append(accuracy)
            val_losses.append(sum(eopoch_losses) / len(eopoch_losses))
            val_accuracies.append(sum(eopoch_accuracies) / len(eopoch_accuracies))
            print(f'epoch: {epoch}')
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax[0, 0].plot(train_losses, label='train loss')
            ax[0, 0].set_title('train loss')
            ax[0, 1].plot(val_losses, label='val loss')
            ax[0, 1].set_title('val loss')
            ax[1, 0].plot(train_accuracies, label='train accuracy')
            ax[1, 0].set_title('train accuracy')
            ax[1, 1].plot(val_accuracies, label='val accuracy')
            ax[1, 1].set_title('val accuracy')
            ax[0, 0].legend()
            # if plots folder doesn't exist, create it
            if not os.path.exists('plots'):
                os.makedirs('plots')
            fig.savefig(f'plots/resnet_experiment_3_{name}_{epoch}.png')
            plt.close()

    train(batch_loop_2, scheduler, name='regular_pretrained')
    train(batch_loop_1, noise_model_scheduler, name='noise_pretrained')




if __name__ == "__main__":
    resnet_experiment_0()
    # resnet_experiment_3()

