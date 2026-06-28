# get_imagenet_dataloader and get_latent_dataloader functions
from torch.utils.data import DataLoader
from .imagenet import get_imagenet_dataloader
from .img_latent_dataset import ImgLatentDataset


def get_latent_dataloader(data_dir, batch_size, num_workers, shuffle=True):
    dataset = ImgLatentDataset(data_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader


