import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm


class MLPCLF(torch.nn.Module):
    def __init__(self, channels=(2 ** 12, 2 ** 6), image_size=(3, 32, 32), num_classes=10):
        super(MLPCLF, self).__init__()
        input_size = np.prod(image_size)
        in_channels = (input_size, *channels)
        out_channels = (*channels, num_classes)
        layers = []

        for i in range(len(in_channels)):
            layers.append(torch.nn.Linear(in_channels[i], out_channels[i]))
            if i < len(in_channels) - 1:
                layers.append(torch.nn.ReLU())

        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.classifier(x.flatten(start_dim=1))
        return x


# def load_img(path):
#     image = Image.open(path).convert("RGB")
#     w, h = image.size
#     image = np.array(image).astype(np.float32) / 255.0
#     image = image[None].transpose(0, 3, 1, 2)
#     image = torch.from_numpy(image)
#     return image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CifarDataset(Dataset):
    def __init__(self, path='/datasets/cifar-10-batches-py/data_batch_1', transform=None):
        self.path = path
        self.transform = transform
        self.data, self.labels = self._make_dataset()

    def _make_dataset(self):
        cifar_dict = unpickle(self.path)
        data = cifar_dict[b'data']
        labels = cifar_dict[b'labels']
        data = data.reshape((10000, 3, 32, 32))
        data = data / 255.0
        data = data.astype(np.float32)
        data = torch.from_numpy(data)
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


def mlp_experiment_2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    models = {
        "model_1": MLPCLF((2**12, 2**12)).to(device),
        # "model_2": MLPCLF((2 ** 12, 2**8, 2**10, 2**6, 2**8, 2**4, 2**6)).to(device),
        # "model_3": MLPCLF((2**12, 2 ** 11, 2 ** 10, 2 ** 9, 2**8, 2**7, 2**6, 2**5, 2**4)).to(device),
        # "model_4": MLPCLF((2**12, 2 ** 11, 2 ** 9, 2 ** 7, 2 ** 5, 2** 7, 2** 9, 2**7, 2**5)).to(device),
        # "model_5": MLPCLF((2**8,)*20).to(device),
        "model_6": MLPCLF((2**13,)).to(device),
        "model_7": MLPCLF((2**12, 2**13)).to(device),
        "model_8": MLPCLF((2**12, 2**6)).to(device),
        "model_9": MLPCLF((2**13, 2**12)).to(device),
        "model_10": MLPCLF((2**14,)).to(device),
        "model_11": MLPCLF((2**15,)).to(device),
        "model_12": MLPCLF((2**16,)).to(device),
        "model_13": MLPCLF((2**10,)).to(device),
        "model_14": MLPCLF((2**9,)).to(device),
        "model_15": MLPCLF((2**8,)).to(device),

    }

    for model_name, model in models.items():
        print(f'Number of parameters for {model_name}: {n_parameters(model)}')

    loss_fun = nn.CrossEntropyLoss()
    learning_rates = [0.001]
    optimizers = {(model_name, lr): torch.optim.Adam(model.parameters(), lr=lr) for model_name, model in models.items() for lr in learning_rates}
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
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    val_dataset = CifarDataset(path='/datasets/cifar-10-batches-py/test_batch')
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4)

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


        fig.legend()
        # if plots folder doesn't exist, create it
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.savefig(f'plots/mlp_experiment_2_{epoch}.png')
        plt.close()
        loss_dict = {"val": val_losses, "train": train_losses}
        torch.save(loss_dict, 'plots/mlp_experiment_2_losses.pt')


if __name__ == "__main__":
    mlp_experiment_2()

    # num_seen = 0
    # num_correct = 0
    # for i in range(10000):
    #     x = data[i]
    #     x = x.to(device)
    #     x = x.unsqueeze(0)
    #     scores = classifier(x)
    #     pred = torch.argmin(scores)
    #     if pred == labels[i]:
    #         num_correct += 1
    #     num_seen += 1
    #     print(f"accuracy: {num_correct / num_seen}")
