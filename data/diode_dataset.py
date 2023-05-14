import os.path
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from .image_folder import make_dataset
from PIL import Image
from functools import partial
import torchvision
import blobfile as bf
from autoencoder.distributions import DiagonalGaussianDistribution
from glob import glob


def get_params(size, resize_size, crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(params, resize_size, crop_size, method=Image.BICUBIC, flip=True, crop=True, totensor=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    if totensor:
        transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    if isinstance(img, torch.Tensor):
        return torch.nn.functional.interpolate(img.unsqueeze(0), size=(target_width, target_width), mode='bicubic',
                                               align_corners=False).squeeze(0)
    else:
        return img.resize((target_width, target_width), method)


def __flip(img, flip):
    if flip:
        if isinstance(img, torch.Tensor):
            return img.flip(-1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def get_flip(img, flip):
    return __flip(img, flip)


class Night2DayDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True, cache_name='cache',
                 disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir)  # get image paths
            self.val_dir = os.path.join(dataroot, 'val')  # get the image directory
            self.val_paths = make_dataset(self.val_dir)  # get image paths
            self.AB_paths = sorted(self.train_paths + self.val_paths)
        else:

            self.test_dir = os.path.join(dataroot, 'test')  # get the image directory
            self.AB_paths = make_dataset(self.test_dir)  # get image paths
        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip

        self.cache_path = os.path.join(dataroot, cache_name + f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        if self.cache is None:
            AB_path = self.AB_paths[index]
            AB = Image.open(AB_path).convert('RGB')
            # split AB image into A and B
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))

            # apply the same transform to both A and B
            params = get_params(A.size, self.resize_size, self.crop_size)

            transform_image = get_transform(params, self.resize_size, self.crop_size, crop=self.random_crop,
                                            flip=self.random_flip)

            A = transform_image(A)
            B = transform_image(B)

            return A, B
        else:

            img = DiagonalGaussianDistribution(self.cache['img'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            cond = DiagonalGaussianDistribution(self.cache['cond'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            params = {'flip': random.random() > 0.5}
            transform_image = transforms.Lambda(lambda img: get_flip(img, params['flip']))
            img = transform_image(img)
            cond = transform_image(cond)

            return img, cond

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class EdgesDataset(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        if train:
            self.train_dir = os.path.join(dataroot, 'train')  # get the image directory
            self.train_paths = make_dataset(self.train_dir)  # get image paths
            self.AB_paths = sorted(self.train_paths)
        else:

            self.test_dir = os.path.join(dataroot, 'val')  # get the image directory
            self.AB_paths = make_dataset(self.test_dir)  # get image paths
        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        params = get_params(A.size, self.resize_size, self.crop_size)

        transform_image = get_transform(params, self.resize_size, self.crop_size, crop=self.random_crop,
                                        flip=self.random_flip)

        A = transform_image(A)
        B = transform_image(B)

        return B, A

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)


class MapsDataset(EdgesDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __getitem__(self, index):
        B, A = super().__getitem__(index)

        return A, B


class FacadesDataset(EdgesDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __getitem__(self, index):
        B, A = super().__getitem__(index)

        return A, B


class COCOStuff(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True, down_sample_img_size=0,
                 cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train2017' if train else 'val2017')
        self.label_root = os.path.join(dataroot,
                                       'annotations/stuff_train2017_pixelmaps_color' if train else 'annotations/stuff_val2017_pixelmaps_color')
        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip

        self.filenames = [l for l in os.listdir(self.image_root) if not l.endswith('.pth')]

        self.cache_path = os.path.join(self.image_root, cache_name + f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        if self.cache is None:
            fn = self.filenames[index]
            img_path = os.path.join(self.image_root, fn)
            label_path = os.path.join(self.label_root, fn[:-4] + '.png')

            with bf.BlobFile(img_path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")

            with bf.BlobFile(label_path, "rb") as f:
                pil_label = Image.open(f)
                pil_label.load()
            pil_label = pil_label.convert("RGB")

            # apply the same transform to both A and B
            params = get_params(pil_image.size, self.resize_size, self.crop_size)

            transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST,
                                            crop=self.random_crop, flip=self.random_flip)
            transform_image = get_transform(params, self.resize_size, self.crop_size, crop=self.random_crop,
                                            flip=self.random_flip)

            cond = transform_label(pil_label)
            img = transform_image(pil_image)

            # if self.down_sample_img:
            #     image_pil = np.array(image_pil).astype(np.uint8)
            #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
            #     down_sampled_image = get_tensor()(down_sampled_image)
            #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
            #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

            #     return image_tensor, data_dict

            return img, cond
        else:

            img = DiagonalGaussianDistribution(self.cache['img'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            cond = DiagonalGaussianDistribution(self.cache['cond'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            params = {'flip': random.random() > 0.5}
            transform_image = transforms.Lambda(lambda img: get_flip(img, params['flip']))
            img = transform_image(img)
            cond = transform_image(cond)

            return img, cond

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)


class DIODE(torch.utils.data.Dataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, dataroot, train=True, img_size=256, random_crop=False, random_flip=True, down_sample_img_size=0,
                 cache_name='cache', disable_cache=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__()
        self.image_root = os.path.join(dataroot, 'train' if train else 'val')
        self.crop_size = img_size
        self.resize_size = img_size

        self.random_crop = random_crop
        self.random_flip = random_flip

        self.filenames = [l for l in os.listdir(self.image_root) if
                          not l.endswith('.pth') and not l.endswith('_depth.png') and not l.endswith('_normal.png')]

        self.cache_path = os.path.join(self.image_root, cache_name + f'_{img_size}.pth')
        if os.path.exists(self.cache_path) and not disable_cache:
            self.cache = torch.load(self.cache_path)
            # self.cache['img'] = self.cache['img'][:256]
            self.scale_factor = self.cache['scale_factor']
            print('Loaded cache from {}'.format(self.cache_path))
        else:
            self.cache = None

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        if self.cache is None:
            fn = self.filenames[index]
            img_path = os.path.join(self.image_root, fn)
            label_path = os.path.join(self.image_root, fn[:-4] + '_normal.png')

            with bf.BlobFile(img_path, "rb") as f:
                pil_image = Image.open(f)
                pil_image.load()
            pil_image = pil_image.convert("RGB")

            with bf.BlobFile(label_path, "rb") as f:
                pil_label = Image.open(f)
                pil_label.load()
            pil_label = pil_label.convert("RGB")

            # apply the same transform to both A and B
            params = get_params(pil_image.size, self.resize_size, self.crop_size)

            transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop=False,
                                            flip=self.random_flip)
            transform_image = get_transform(params, self.resize_size, self.crop_size, crop=False, flip=self.random_flip)

            cond = transform_label(pil_label)
            img = transform_image(pil_image)

            # if self.down_sample_img:
            #     image_pil = np.array(image_pil).astype(np.uint8)
            #     down_sampled_image = self.down_sample_img(image=image_pil)["image"]
            #     down_sampled_image = get_tensor()(down_sampled_image)
            #     # down_sampled_image = transforms.ColorJitter(brightness = [0.85,1.15], contrast=[0.95,1.05], saturation=[0.95,1.05])(down_sampled_image)
            #     data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}

            #     return image_tensor, data_dict

            return {'A': img, 'B': cond, 'A_paths': img_path, 'B_paths': label_path}
        else:

            img = DiagonalGaussianDistribution(self.cache['img'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            cond = DiagonalGaussianDistribution(self.cache['cond'][index].unsqueeze(0)).sample().squeeze(
                0) * self.scale_factor
            params = {'flip': random.random() > 0.5}
            transform_image = transforms.Lambda(lambda img: get_flip(img, params['flip']))
            img = transform_image(img)
            cond = transform_image(cond)

            return img, cond

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.cache is not None:
            return len(self.cache['img'])
        else:
            return len(self.filenames)


class PriorCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, sigma_max=80.0, pred_mode='uncond', **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_max = sigma_max
        self.pred_mode = pred_mode

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)  # img in [0, 1]
        if self.pred_mode.startswith('corr-uncond'):
            prior = torch.randn_like(img) * self.sigma_max / 2 + 0.5
        else:
            prior = img + torch.randn_like(img) * self.sigma_max / 2
        return img, prior

    def prior(self, shape, device):
        return torch.randn(shape, device=device) * self.sigma_max
