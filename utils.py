import os
import torch
import numpy as np
from PIL import Image
from torchvision.models import vgg16


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.attacked_images = sorted([x for x in os.listdir(root_dir) if not 'ori' in x])
        self.original_images = [x.split('.')[0] + '_origin.png' for x in self.attacked_images]

    def __getitem__(self, index):
        attacked_image = Image.open(os.path.join(self.root_dir, self.attacked_images[index]))
        original_image = Image.open(os.path.join(self.root_dir, self.original_images[index]))

        if attacked_image.mode != 'RGB':
            attacked_image = attacked_image.convert("RGB")
        if original_image.mode != 'RGB':
            original_image = original_image.convert("RGB")

        if self.transform:
            attacked_image = self.transform(attacked_image)
            original_image = self.transform(original_image)

        return attacked_image * 2 - 1, original_image * 2 - 1

    def __len__(self):
        return len(self.attacked_images)


def vgg16_pretrained(model_path):
    model = vgg16(pretrained=False)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model
