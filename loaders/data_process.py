import os
from utils.tools import get_name
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class DataSplit():
    def __init__(self, meta=True):
        self.meta = meta
        if meta:
            pp = "building extra"
        else:
            pp = "building_jiangxi"
        self.root = "/media/wbw/a7f02863-b441-49d0-b546-6ef6fefbbc7e/" + pp
        self.cat = ['Temples', 'Ancestral Hall', 'Bridge', 'The Plaque', 'Pavilion', 'Modern Historic', 'Loft', 'Residential Buildings', 'Theatres', 'Tower']

    def get_all_imgs(self):
        imgs = []
        folders = get_name(self.root)
        for folder in folders:
            current_imgs = get_name(os.path.join(self.root, folder), mode_folder=False)
            if not self.meta and folder == 'Residential Buildings':
                current_imgs, pp = train_test_split(current_imgs, train_size=0.2, random_state=1)
            for current_img in current_imgs:
                imgs.append([os.path.join(self.root, folder, current_img), self.cat.index(folder)])
        return imgs

    def get_split_data(self):
        all_imgs = self.get_all_imgs()
        if self.meta:
            train_list, val_list = train_test_split(all_imgs, train_size=0.9, random_state=2)
            return {"train": train_list, "val": val_list}, self.cat
        else:
            return {"train": all_imgs, "val": []}, self.cat


def get_train_transformations(args):
    aug_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation([-50, 50]),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                ], p=0.5),
                transforms.Resize([args.img_size, args.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
                ]
    return transforms.Compose(aug_list)


def get_val_transformations(args):
    aug_list = [transforms.Resize([args.img_size, args.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])
                ]
    return transforms.Compose(aug_list)


def get_train_dataloader(args, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=args.num_workers,
            batch_size=args.batch_size, pin_memory=False, drop_last=True, shuffle=True)


def get_val_dataloader(args, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=args.num_workers,
            batch_size=args.batch_size, pin_memory=False, drop_last=False, shuffle=False)


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, args, phase, transform, meta=True):
        self.all_data, self.cat = DataSplit(meta).get_split_data()
        self.all_data = self.all_data[phase]
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item_id):
        image_root = self.all_data[item_id][0]
        image = Image.open(image_root).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        label = self.all_data[item_id][1]
        label = torch.from_numpy(np.array(label))
        return image, label, image_root
