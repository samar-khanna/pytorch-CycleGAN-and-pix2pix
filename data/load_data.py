import torch
import torchvision

import torchvision.transforms as transforms

import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
# import imageio

import math
import random
import torch.distributed as dist

from torch.utils.data.sampler import Sampler

from torch.utils.data import DataLoader, Dataset


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


class UniformDequant(object):
    def __call__(self, x):
        return x + torch.rand_like(x) / 256


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.
    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, seed=0, repeated_aug=0, filling=False,
                 shuffle=True):
        # from torchvision.models import ResNet50_Weights
        # RA sampler: https://github.com/pytorch/vision/blob/5521e9d01ee7033b9ee9d421c1ef6fb211ed3782/references/classification/sampler.py

        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size

        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[
                                 :(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(
                    self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())

        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices


def load_data(
        data_dir,
        dataset,
        batch_size,
        image_size,
        diffusion,
        deterministic=False,
        include_test=False,
        seed=42,
        num_workers=2,
):
    # Compute batch size for this worker.
    root = data_dir

    if dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=
        transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))
        valset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.Resize(32), transforms.ToTensor()]))

        # train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        # test_X, test_y = valset.data[:,None] / 255, valset.targets
        n_classes = 10
    elif dataset == 'FMNIST':
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=
        transforms.Compose([transforms.ToTensor()]))
        valset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.ToTensor()]))

        # train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        # test_X, test_y = valset.data[:,None] / 255, valset.targets
        n_classes = 10
    elif dataset == 'cifar10':
        from .aligned_dataset import PriorCIFAR10
        trainset = PriorCIFAR10(root=root, train=True, download=True, transform=
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ]), sigma_max=diffusion.sigma_max, pred_mode=diffusion.pred_mode)
        valset = PriorCIFAR10(root=root, train=True, download=True, transform=
        transforms.Compose([
            transforms.ToTensor(), ]), sigma_max=diffusion.sigma_max, pred_mode=diffusion.pred_mode)

        # train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        # train_y = torch.tensor(trainset.targets, dtype=int)

        # test_X = torch.tensor(valset.data).permute(0,3,1,2) / 255
        # test_y = torch.tensor(valset.targets, dtype=int)
        n_classes = 10
    elif dataset == 'afhqv2' or dataset == 'ffhq':
        # from .afhq import PriorAFHQv2
        # trainset = PriorAFHQv2(path=os.path.join(root, 'afhqv2-64x64.zip'), sigma_max=diffusion.sigma_max,
        #                       xflip=True)
        # valset = PriorAFHQv2(path=os.path.join(root, 'afhqv2-64x64.zip'), sigma_max=diffusion.sigma_max,
        #                       xflip=False)
        from .afhq import PriorAFHQv2Folder
        trainset = PriorAFHQv2Folder(root, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), ]), sigma_max=diffusion.sigma_max, pred_mode=diffusion.pred_mode)
        valset = PriorAFHQv2Folder(root, transform=transforms.Compose([
            transforms.ToTensor(), ]), sigma_max=diffusion.sigma_max, pred_mode=diffusion.pred_mode)


    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=
        transforms.Compose([transforms.ToTensor(), ]))
        valset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.ToTensor()]))

        # train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        # train_y = torch.tensor(trainset.targets, dtype=int)

        # test_X = torch.tensor(valset.data).permute(0,3,1,2) / 255
        # test_y = torch.tensor(valset.targets, dtype=int)
        n_classes = 100
    elif dataset == 'TinyImageNet':
        from .imagenet import TinyImageNetDataset
        trainset = TinyImageNetDataset(root_dir=root, mode='train', download=False, transform=
        transforms.Compose([transforms.ToTensor(), ]))
        valset = TinyImageNetDataset(root_dir=root, mode='val', download=False, transform=
        transforms.Compose([transforms.ToTensor()]))

        n_classes = 200

    elif dataset == 'ImageNet':
        from .imagenet import ImageNetDataset, ImageNet128Dataset
        if image_size == 128:
            trainset = ImageNet128Dataset(root=root, is_train=True, transform=transforms.RandomHorizontalFlip(), )
            valset = ImageNet128Dataset(root=root, is_train=False, transform=transforms.RandomHorizontalFlip(), )

        else:
            transform = [
                transforms.RandomResizedCrop(image_size, scale=(0.67, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
            trainset = ImageNetDataset(root=root, train=True, transform=
            transforms.Compose(transform))
            if image_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(image_size / crop_pct)
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC, ),
                transforms.CenterCrop(image_size),
            ])
            valset = ImageNetDataset(root=root, train=False, transform=test_transform)

            n_classes = 1000

    elif dataset == 'toy':

        mix = torch.distributions.Categorical(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        comp = torch.distributions.Normal(torch.tensor([-10., 0., 10., 20.]),
                                          torch.tensor([0.001, 0.001, 0.001, 0.001]))
        gmm = torch.distributions.MixtureSameFamily(mix, comp)
        train_X = gmm.sample((2000, 1, 1, 1))
        train_Y = train_X - 10

        test_X = gmm.sample((200, 1, 1, 1))
        test_Y = test_X - 10

        # create torch tensordataset
        trainset = torch.utils.data.TensorDataset(train_X, train_Y)
        valset = torch.utils.data.TensorDataset(test_X, test_Y)


    elif dataset == 'night2day':

        from .diode_dataset import Night2DayDataset
        trainset = Night2DayDataset(dataroot=root, train=True, img_size=image_size,
                                    random_crop=True, random_flip=True, )

        valset = Night2DayDataset(dataroot=root, train=True, img_size=image_size,
                                  random_crop=False, random_flip=False, disable_cache=True)
        if include_test:
            testset = Night2DayDataset(dataroot=root, train=False, img_size=image_size,
                                       random_crop=False, random_flip=False, disable_cache=True)
    elif dataset == 'maps':
        from .diode_dataset import MapsDataset
        trainset = MapsDataset(dataroot=root, train=True, img_size=image_size,
                               random_crop=True, random_flip=True)

        valset = MapsDataset(dataroot=root, train=True, img_size=image_size,
                             random_crop=False, random_flip=False)

    elif dataset == 'edges2handbags':

        from .diode_dataset import EdgesDataset
        trainset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                                random_crop=True, random_flip=True)

        valset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                              random_crop=False, random_flip=False)
        if include_test:
            testset = EdgesDataset(dataroot=root, train=False, img_size=image_size,
                                   random_crop=False, random_flip=False)
    elif dataset == 'facades':

        from .diode_dataset import FacadesDataset
        trainset = FacadesDataset(dataroot=root, train=True, img_size=image_size,
                                  random_crop=True, random_flip=True)

        valset = FacadesDataset(dataroot=root, train=False, img_size=image_size,
                                random_crop=False, random_flip=False)

    elif dataset == 'coco':

        from .diode_dataset import COCOStuff
        trainset = COCOStuff(dataroot=root, train=True, img_size=image_size,
                             random_crop=True, random_flip=True)

        valset = COCOStuff(dataroot=root, train=True, img_size=image_size,
                           random_crop=False, random_flip=False, disable_cache=True)

    elif dataset == 'diode':

        from .diode_dataset import DIODE
        trainset = DIODE(dataroot=root, train=True, img_size=image_size,
                         random_crop=True, random_flip=True, disable_cache=True)

        valset = DIODE(dataroot=root, train=True, img_size=image_size,
                       random_crop=False, random_flip=False, disable_cache=True)

        if include_test:
            valset = DIODE(dataroot=root, train=False, img_size=image_size,
                           random_crop=False, random_flip=False)
    elif dataset == 'sketch':
        import glob
        import PIL
        class Sketch(torch.utils.data.Dataset):
            def __init__(self):
                path = "/alex/data/coco/train2017/*.jpg"
                self.files = glob.glob(path)

                self.save_dir = "/alex/data/coco/annotations/stuff_train2017_sketch"

            def __getitem__(self, index):
                inimg = PIL.Image.open(self.files[index]).convert('RGB').resize((256, 256))
                tenInput = torch.FloatTensor(np.ascontiguousarray(
                    np.array(inimg)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
                return tenInput, self.files[index]

            def __len__(self):
                return len(self.files)

        trainset = valset = Sketch()

    loader = DataLoader(
        dataset=trainset, num_workers=num_workers, pin_memory=True,
        batch_size=batch_size, shuffle=True
        # batch_sampler=DistInfiniteBatchSampler(
        #     dataset_len=len(trainset), glb_batch_size=batch_size * dist.get_world_size(), seed=seed,
        #     shuffle=not deterministic, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        # )
    )

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

    if include_test:

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            testset, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size,
            sampler=sampler, num_workers=num_workers, drop_last=False)

        return loader, val_loader, test_loader
    else:
        return loader, val_loader
