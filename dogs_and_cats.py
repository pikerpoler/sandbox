import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Conv2d, Linear, MaxPool2d, AdaptiveAvgPool1d
from torch.nn.functional import relu, dropout
from ImageLoader import ImageLoader
from torch.utils.data import DataLoader
import platform




TRAIN_SET_PATH = "training_set"
# train_data = DataLoader("/Users/nadav.nissim/Downloads/archive/training_set/training_set", batch_size=10, num_workers=1, shuffle=True)
# test_data = DataLoader("/Users/nadav.nissim/Downloads/archive/test_set/test_set")


DEVICE = "cuda" if platform.system() == "Linux" else "cpu"

device = torch.device(DEVICE)

class Network(Module):

    def __init__(self):
        super(Network, self).__init__()
    #     self.conv_1 = Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5))
    #     self.conv_2 = Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5))
    #     self.conv_3 = Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5))
    #
    #     self.maxPooling = MaxPool2d(kernel_size=4)
    #     self.adPooling = AdaptiveAvgPool1d(256)
    #
    #     self.fc_1 = Linear(in_features=256, out_features=128)
    #     self.fc_2 = Linear(in_features=128, out_features=64)
    #     self.fc_3 = Linear(in_features=64, out_features=32)
    #     self.out = Linear(in_features=32, out_features=2)
    #     self.softmax = torch.nn.Softmax(dim=1)
    #
    # def forward(self, x):
    #     x = self.conv_1(x)
    #     x = self.maxPooling(x)
    #     x = relu(x)
    #
    #     x = self.conv_2(x)
    #     x = self.maxPooling(x)
    #     x = relu(x)
    #
    #     x = self.conv_3(x)
    #     x = self.maxPooling(x)
    #     x = relu(x)
    #
    #     x = dropout(x)
    #     x = x.view(1, x.size()[0], -1)
    #     x = self.adPooling(x).squeeze()
    #
    #     x = self.fc_1(x)
    #     x = relu(x)
    #
    #     x = self.fc_2(x)
    #     x = relu(x)
    #
    #     x = self.fc_3(x)
    #     x = relu(x)
    #
    #     return self.softmax(relu(self.out(x)))
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# hyper parameters:
num_epochs = 4
batch_size = 10
learning_rate = 0.001

network = Network()

# imageLoader = ImageLoader(TRAIN_SET_PATH)
# dataLoader = DataLoader(imageLoader, batch_size=batch_size, shuffle=True)
# data = iter(dataLoader)
# images = next(data)
# out = network(images[0])

dataLoader = DataLoader(ImageLoader(TRAIN_SET_PATH), batch_size= batch_size, shuffle=True)

model = network.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.09)


n_total_steps = len(dataLoader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataLoader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f} %')
            pred = torch.argmax(outputs, dim=1)
            gt = torch.argmax(labels, dim=1)
            acc = sum(abs(pred - gt)) / 10
            print(f'accuracy = {acc}')


print("done")


#
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     # n_class_correct = [0, 0]
#     # n_class_samples = [0, 0]
#
#     for images, labels in testDataLoader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#
#         _, predicted = torch.max(outputs, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#         # for i in range(batch_size):
#         #     label = labels[i]
#         #     pred = predicted[i]
#         #     if (label == pred):
#         #         n_class_correct[label] += 1
#         #     n_class_samples += 1
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network: {acc} %')
#
