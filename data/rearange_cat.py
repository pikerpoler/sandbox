import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import PIL.Image as Image


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



def main():
    print('rearanging cat data')
    data_path = '/Users/nadav.nissim/Documents/Coding/data/cat1data'
    new_data_path = '/Users/nadav.nissim/Documents/Coding/data/cat_rearranged'

    # create cat_rearranged folder if it doesn't exist
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)
    # create bmp and png folders if they don't exist
    if not os.path.exists(os.path.join(new_data_path, 'bmp')):
        os.mkdir(os.path.join(new_data_path, 'bmp'))
    if not os.path.exists(os.path.join(new_data_path, 'png')):
        os.mkdir(os.path.join(new_data_path, 'png'))


    dataset = CatPresentSingle(data_path, transform=None)
    for i, (img, label) in enumerate(dataset):
        if label[0] != 0:
            # save image to bmp folder and png folder
            img.save(os.path.join(new_data_path, "bmp", f'bmp_{i}_label:{label[0]}.bmp'))
            img.save(os.path.join(new_data_path, "png", f'png_{i}_label:{label[0]}.png'))



if __name__ == '__main__':
    main()