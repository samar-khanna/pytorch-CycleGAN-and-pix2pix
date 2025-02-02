"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data


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

        from .aligned_dataset import Night2DayDataset
        trainset = Night2DayDataset(dataroot=root, train=True, img_size=image_size,
                                    random_crop=True, random_flip=True, )

        valset = Night2DayDataset(dataroot=root, train=True, img_size=image_size,
                                  random_crop=False, random_flip=False, disable_cache=True)
        if include_test:
            testset = Night2DayDataset(dataroot=root, train=False, img_size=image_size,
                                       random_crop=False, random_flip=False, disable_cache=True)
    elif dataset == 'maps':
        from .aligned_dataset import MapsDataset
        trainset = MapsDataset(dataroot=root, train=True, img_size=image_size,
                               random_crop=True, random_flip=True)

        valset = MapsDataset(dataroot=root, train=True, img_size=image_size,
                             random_crop=False, random_flip=False)

    elif dataset == 'edges2handbags':

        from .aligned_dataset import EdgesDataset
        trainset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                                random_crop=True, random_flip=True)

        valset = EdgesDataset(dataroot=root, train=True, img_size=image_size,
                              random_crop=False, random_flip=False)
        if include_test:
            testset = EdgesDataset(dataroot=root, train=False, img_size=image_size,
                                   random_crop=False, random_flip=False)
    elif dataset == 'facades':

        from .aligned_dataset import FacadesDataset
        trainset = FacadesDataset(dataroot=root, train=True, img_size=image_size,
                                  random_crop=True, random_flip=True)

        valset = FacadesDataset(dataroot=root, train=False, img_size=image_size,
                                random_crop=False, random_flip=False)

    elif dataset == 'coco':

        from .aligned_dataset import COCOStuff
        trainset = COCOStuff(dataroot=root, train=True, img_size=image_size,
                             random_crop=True, random_flip=True)

        valset = COCOStuff(dataroot=root, train=True, img_size=image_size,
                           random_crop=False, random_flip=False, disable_cache=True)

    elif dataset == 'diode':

        from .aligned_dataset import DIODE
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
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(trainset), glb_batch_size=batch_size * dist.get_world_size(), seed=seed,
            shuffle=not deterministic, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        )
    )