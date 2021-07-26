#!/usr/bin/env python
# coding: utf-8

# We acknowledge the following repo from where we adopt the GradCAM module
# https://github.com/jacobgil/pytorch-grad-cam

from collections import OrderedDict, Sequence
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import cv2
from utils import ImageDataset, vgg16_pretrained
from torchvision import transforms
from torchvision.models import vgg16
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input

class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def check_layers(self, layer):
        instance_list = [nn.Linear, nn.ReLU, nn.Conv2d, nn.MaxPool2d,
                         nn.AvgPool2d, nn.Dropout, nn.BatchNorm2d,
                         nn.AdaptiveAvgPool2d]
        for instance in instance_list:
            if isinstance(layer, instance):
                return True
        return False

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model.named_modules():
            if not self.check_layers(module):
                continue

            if isinstance(module, nn.Linear):
                x = torch.flatten(x, 1)
            x = module(x)

            if name == self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            # if name in ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_3', 'relu_5_3'] \
            #         or name in ['5', '12', '19']:
            #     x = F.max_pool2d(x, 2, 2)
        return outputs, x


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        # output = output.view(output.size(0), -1)
        # output = self.model.relu6(self.model.fc6(output))
        # output = F.dropout(output, 0.5)
        # output = self.model.relu7(self.model.fc7(output))
        # output = F.dropout(output, 0.5)
        # output = self.model.fc8(output)
        return target_activations, output


class TaintRadar:
    def __init__(self, model, use_cuda, device, layer="relu_5_3"):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to(device)
            self.device = device

        self.extractor = ModelOutputs(self.model, layer)
        self.__registered_image = []

    def forward(self, input):
        return self.model(input)

    def mask_post_process(self, mask, bin_thresh, input_shape):
        mask = np.maximum(mask, 0)
        mask = cv2.resize(mask, input_shape)
        if np.max(mask) != 0:
            mask = mask / np.max(mask)
        mask[mask < bin_thresh * np.max(mask)] = 0
        mask[mask >= bin_thresh * np.max(mask)] = 1
        return mask

    def critical_region_estimation(self, features, output, class_idx, div, input_shape, bin_thresh=0.15):
        self.model.zero_grad()

        prob = F.softmax(output / div, dim=-1)
        x = prob
        y = torch.tensor(class_idx).unsqueeze(dim=0).cuda()
        loss = nn.CrossEntropyLoss()(x, y)

        loss.backward(retain_graph=True)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        mask = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            mask += w * target[i, :, :]

        mask = self.mask_post_process(mask.copy(), bin_thresh, input_shape)
        return mask

    def generate_negative_sub_mask(self, features, output, class_index, input_shape, bin_thresh=0.15, wo_softmax=True):
        if wo_softmax:
            output[:, class_index].backward(retain_graph=True)
        else:
            prob = F.softmax(output, dim=1)
            prob[:, class_index].backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]

        mask = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            mask += w * target[i, :, :]

        mask = self.mask_post_process(-1 * mask.copy(), bin_thresh, input_shape)  # negative mask

        return mask

    def negative_mask_generation(self, features, output, class_indices, input_shape, bin_thresh=0.15, wo_softmax=True):
        self.model.zero_grad()
        final_mask = np.ones(input_shape)

        for idx in class_indices:
            mask = self.generate_negative_sub_mask(features, output, idx, input_shape, bin_thresh, wo_softmax)
            self.__register_fig(mask, str(idx))
            final_mask = final_mask * mask

        return final_mask

    def forward(self, input):
        if self.cuda:
            features, output = self.extractor(input.to(self.device))
        else:
            features, output = self.extractor(input)
        return features, output

    def content_filling(self, input, mask, filling_content=None):
        if filling_content is None:
            torch.manual_seed(1)
            filling_content = torch.randn_like(input[0])
        mask = torch.from_numpy(np.tile(np.expand_dims(mask, 0), (3, 1, 1)))
        surgical_removal = input.clone()
        surgical_removal[0, mask != 0] = filling_content[mask != 0]
        return surgical_removal

    def process(self, input, top_k, delta_r, filling_content=None, preprocess_input=None, sign=1, gen_negative=False,
                div=2, verbose=True):
        self.__register_fig(np.transpose(input[0].cpu().numpy(), (1, 2, 0)), 'o_i')
        features_before, output_before = self.forward(input)

        original_idx = np.argmax(output_before.cpu().data.numpy())
        critical_region = self.critical_region_estimation(features=features_before, output=output_before,
                                                          class_idx=original_idx, div=div,
                                                          input_shape=input.shape[-2:])
        self.__register_fig(critical_region, 'critical')
        intermediate_input = self.content_filling(input, critical_region, filling_content).requires_grad_(False)
        self.__register_fig(np.transpose(intermediate_input[0].cpu().numpy(), (1, 2, 0)), 'i_i')

        features_after, output_after = self.forward(intermediate_input)
        self.model.zero_grad()
        features_before, output_before = self.forward(input)
        delta = output_after - output_before
        sorted_delta = np.argsort(delta[0].detach().cpu().numpy())
        final_mask = self.negative_mask_generation(features=features_before, output=output_before,
                                                   class_indices=torch.from_numpy(sorted_delta[-top_k:]),
                                                   input_shape=input.shape[-2:])
        final_input = self.content_filling(input, final_mask, filling_content)
        self.__register_fig(final_mask, 'final')
        self.__register_fig(np.transpose(final_input[0].cpu().numpy(), (1, 2, 0)), 'f_i')

        _, output_final = self.forward(final_input)
        ranking_delta = np.argsort(output_final[0].detach().cpu().numpy(), ).tolist().index(original_idx)
        self.__plot_registered(verbose)
        return ranking_delta < delta_r

    def __register_fig(self, img, title):
        self.__registered_image.append([img, title])

    def __plot_registered(self, verbose):
        if verbose:
            im_num = len(self.__registered_image)
            cols = np.floor(np.sqrt(im_num))
            rows = np.floor(im_num / cols)

            if im_num % cols != 0:
                rows += 1
            rowwise_len = rows * 3 + 1
            colwise_len = cols * 3 + 1
            plt.figure(1, (rowwise_len, colwise_len))
            for i in range(im_num):
                ax = plt.subplot(cols, rows, i + 1)
                ax.imshow(self.__registered_image[i][0])
                if self.__registered_image[i][1]:
                    ax.title.set_text(self.__registered_image[i][1])
            plt.show()
        self.__registered_image = []


if __name__ == "__main__":
    model = vgg16_pretrained(model_path="models/vgg16.pth").cuda().eval()
    print(model)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_path = "images"
    dataset = ImageDataset(root_dir=image_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)
    taint_radar = TaintRadar(model, use_cuda=True, device='cuda:0', layer='features.29')

    result = []
    for attacked_image, original_image in tqdm(loader):
        result.append(taint_radar.process(attacked_image, 9, 2))

    # print(np.sum(result), np.sum(result) / len(result))
    # for attacked_image, original_image in loader:
        # _, output = taint_radar.forward(attacked_image.cuda())
        # print(output.shape, np.argmax(output.detach().cpu().numpy(), axis=-1))
        # original_image = original_image[:, (2, 1, 0), :, :]
        # output = model(original_image.cuda())
        # print(output.shape, np.argmax(output.detach().cpu().numpy(), axis=-1))
