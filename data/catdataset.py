import os
import pandas as pd
import PIL.Image as Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CatPresentSingle(Dataset):
    def __init__(self, data_root, split='train',  xslx_name="cat1.xlsx", transform=None):
        self.data_root = data_root
        self.transform = transform
        xslx_path = os.path.join(data_root, xslx_name)
        self.df = pd.read_excel(xslx_path)
        self.df['subfolder'] = self.df['image path'].apply(lambda x: x.split('\\')[-2])
        self.df['image num'] = self.df['image path'].apply(lambda x: int(x.split('\\')[-1].split('.')[1]))
        self.df = self.df.sort_values(by=['subfolder', 'image num'], ignore_index=True)
        self._apply_split(split)


    def __getitem__(self, idx):
        subfolder = self.df.loc[idx, 'subfolder']
        image_num = self.df.loc[idx, 'image num']
        img_path = os.path.join(self.data_root, subfolder, f"img.{image_num}.bmp")
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        return img, torch.Tensor(self._extract_labels(idx))

    def __len__(self):
        return len(self.df)

    def _extract_labels(self, idx):
        return self.df.loc[idx]['label'],\
               self.df.loc[idx]['x1'], self.df.loc[idx]['y1'],\
               self.df.loc[idx]['x2'], self.df.loc[idx]['y2']

    def _apply_split(self, split):
        unique_subfolders = self.df['subfolder'].unique()
        if split == 'train':
            subfolders = unique_subfolders[:int(len(unique_subfolders) * 0.6)]
        elif split == 'val':
            subfolders = unique_subfolders[int(len(unique_subfolders) * 0.6):int(len(unique_subfolders) * 0.8)]
        elif split == 'test':
            subfolders = unique_subfolders[int(len(unique_subfolders) * 0.8):]
        else:
            raise ValueError(f"split must be one of train, val, test, got {split}")
        self.df = self.df[self.df['subfolder'].isin(subfolders)]
        self.df = self.df.reset_index(drop=True)


class CatFutureWindow(Dataset):
    def __init__(self, data_root, window_size, binary=False, ignore_present=False, transform=None):
        self.data_root = data_root
        self.window_size = window_size
        self.binary = binary
        self.ignore_present = ignore_present
        self.transform = transform


    def __getitem__(self, item):
        return self.dataset[item:item+self.window_size]

    def __len__(self):
        return len(self.dataset) - self.window_size


if __name__ == '__main__':

    # data_path = '/Users/nadav.nissim/Desktop/cat1data/'
    data_path = '/home/nadav.nissim/data/xray/cat1data'
    dataset = CatPresentSingle(data_path, transform=transforms.ToTensor())
    print(dataset[0])
    dl = DataLoader(dataset, batch_size=3, shuffle=True)

    for i, (img, label) in enumerate(dl):
        print(img.size(), label[:,[0]])
        if i == 10:
            break


