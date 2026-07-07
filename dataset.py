from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
import torch.multiprocessing
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_transforms(size, isize, mean_train=None, std_train=None):
    mean_train = [0.485, 0.456, 0.406] if mean_train is None else mean_train
    std_train = [0.229, 0.224, 0.225] if std_train is None else std_train
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types, self.cls_names = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.cls_idx = 0

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        tot_cls_names = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.JPG") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))
            cls_names = self.img_path.split('/')[-2]
            tot_cls_names.extend([cls_names] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return np.array(img_tot_paths), np.array(gt_tot_paths), np.array(tot_labels), np.array(tot_types), np.array(tot_cls_names)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type, cls_name = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx], self.cls_names[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, cls_name

class VisADataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, class_names=None, only_normal_test=False):
        self.root = root
        self.transform = transform
        self.target_transform = gt_transform
        self.phase = phase
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[phase]

        if class_names:
            self.class_names = [class_names]
            for classes in self.class_names:
                assert classes in list(meta_info.keys()), f"Class {classes} is not in dataset {root}"
        else:
            self.class_names = list(meta_info.keys())
        for classes in self.class_names:
            self.data_all.extend(meta_info[classes])

        if only_normal_test and phase == 'test':
            self.data_all = [d for d in self.data_all if d['anomaly'] == 0]
        self.length = len(self.data_all)

        self.class_to_idx = {}
        for k, index in zip(self.class_names, range(len(self.class_names))):
            self.class_to_idx[k] = index

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img_path_ = os.path.join(self.root, img_path)

        img = Image.open(img_path_).convert('RGB')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        if self.phase == 'train':
            return img, anomaly

        if anomaly == 0:
            img_mask = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            # transforms
            img_mask = self.target_transform(
                img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return img, img_mask, anomaly, os.path.join(self.root, img_path), cls_name

class RealIADDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, transform, gt_transform, phase):
        self.img_path = os.path.join(root, category)
        self.transform = transform
        self.gt_transform = gt_transform
        self.phase = phase

        json_path = os.path.join(root, 'realiad_jsons', category + '.json')
        with open(json_path) as file:
            class_json = file.read()
        class_json = json.loads(class_json)

        self.img_paths, self.gt_paths, self.labels, self.types, self.cls_names = [], [], [], [], []

        data_set = class_json[phase]
        for sample in data_set:
            self.img_paths.append(os.path.join(root, category, sample['image_path']))
            self.cls_names.append(category)
            label = sample['anomaly_class'] != 'OK'
            if label:
                self.gt_paths.append(os.path.join(root, category, sample['mask_path']))
            else:
                self.gt_paths.append(None)
            self.labels.append(label)
            self.types.append(sample['anomaly_class'])

        self.img_paths = np.array(self.img_paths)
        self.gt_paths = np.array(self.gt_paths)
        self.labels = np.array(self.labels)
        self.types = np.array(self.types)
        self.cls_names = np.array(self.cls_names)
        self.cls_idx = 0

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type, cls_name = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx], self.cls_names[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label

        if label == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_path, cls_name

class BTADDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, class_names=None):
        self.root = root
        self.transform = transform
        self.target_transform = gt_transform
        self.phase = phase
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        meta_info = meta_info[phase]

        if class_names:
            self.class_names = [class_names]
            for classes in self.class_names:
                assert classes in list(meta_info.keys()), f"Class {classes} is not in dataset {root}"
        else:
            self.class_names = list(meta_info.keys())

        for classes in self.class_names:
            self.data_all.extend(meta_info[classes])
        self.length = len(self.data_all)

        self.class_to_idx = {}
        for k, index in zip(self.class_names, range(len(self.class_names))):
            self.class_to_idx[k] = index

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img_path_ = os.path.join(self.root, img_path)

        img = Image.open(img_path_).convert('RGB')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        if self.phase == 'train':
            return img, anomaly

        if anomaly == 0:
            img_mask = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = torch.zeros([1, img.size()[-2], img.size()[-2]])
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            # transforms
            img_mask = self.target_transform(
                img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        return img, img_mask, anomaly, os.path.join(self.root, img_path), cls_name
