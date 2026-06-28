import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def get_imagenet_dataset(
    data_dir,
    image_size,
    transform_mean=[0., 0., 0.],
    transform_std=[1, 1, 1],
    random_flip_prob=0.,
):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
        transforms.RandomHorizontalFlip(p=random_flip_prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std, inplace=True)
    ])
    dataset = ImageFolder(data_dir, transform=transform)
    return dataset


def get_imagenet_dataloader(
    data_dir,
    image_size,
    batch_size,
    num_workers,
    shuffle=True,
    transform_mean=[0., 0., 0.],
    transform_std=[1, 1, 1],
    random_flip_prob=0.,
):
    dataset = get_imagenet_dataset(
        data_dir=data_dir,
        image_size=image_size,
        transform_mean=transform_mean,
        transform_std=transform_std,
        random_flip_prob=random_flip_prob,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader