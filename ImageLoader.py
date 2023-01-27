import torch
from torchvision.datasets import ImageFolder
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms



PATH_TO_TRAINING_SET = "training_set"

train_dataset = ImageFolder(PATH_TO_TRAINING_SET)
# test_dataset = ImageFolder("~/sandbox/training_set")

default_transform = transforms.Compose([
    transforms.Resize(200),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# train_data, test_data, train_labels, test_labels = train_test_split(train_dataset.imgs, train_dataset.targets, test_size=0.01)
# print(train_data, train_labels)


class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = ImageFolder(dataset)
        if transform:
            self.transform = transform
        else:
            self.transform = default_transform

    def checkChannel(self):
        datasetRGB = []
        for index in range(len(self.dataset)):
            if self.dataset[index][0].getbands() == ('R', 'G', 'B'):
                datasetRGB.append(self.dataset[index])
        if len(datasetRGB) != len(self.dataset):
            print("ERROR")
        else:
            print("FINE")

    def getResizedImage(self,item):
        image = self.dataset[item][0]
        _, _, width, height = image.getbbox()
        factor = (0, 0, width, width) if width > height else (0, 0, height, height)
        return image.crop(factor)

    def __getitem__(self, item):
        image = self.getResizedImage(item)

        black = torch.zeros(3, 200, 200)
        white = 250 * torch.ones(3, 200, 200)
        tag = self.dataset[item][1]
        if tag == 0:
            return black, torch.tensor([0.0, 1.0])
        else:
            return white, torch.tensor([1.0, 0.0])
        return self.transform(image), self.dataset[item][1]

    def __len__(self):
        return len(self.dataset)



# imageLoader = ImageLoader(PATH_TO_TRAINING_SET)


# imageLoader.checkChannel()
# print(imageLoader[0][0].size())

# daraLoader = DataLoader(imageLoader, batch_size=10, shuffle=True)
# data = iter(daraLoader)
# d = next(data)
# print(d[0].size())



