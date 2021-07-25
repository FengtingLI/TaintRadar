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
# from tqdm import tqdm
import cv2


class FeatureExtractor:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            if "fc" in name:
                break

            x = module(x)

            if name == self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
            if name in ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_3', 'relu_5_3'] \
                    or name in ['5', '12', '19']:
                x = F.max_pool2d(x, 2, 2)
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
        output = output.view(output.size(0), -1)
        output = self.model.relu6(self.model.fc6(output))
        output = F.dropout(output, 0.5)
        output = self.model.relu7(self.model.fc7(output))
        output = F.dropout(output, 0.5)
        output = self.model.fc8(output)
        return target_activations, output


class FengtingCAM:
    def __init__(self, model, use_cuda, device, layer='relu_5_3'):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to(device)
            self.device = device

        self.extractor = ModelOutputs(self.model, layer)

    def forward(self, input):
        return self.model(input)

    def mask_post_process(self, mask, bin_thresh, input_shape):
        mask = np.maximum(mask, 0)
        mask = cv2.resize(mask, input_shape)
        mask = mask / np.max(mask)
        mask[mask < bin_thresh * np.max(mask)] = 0
        mask[mask >= bin_thresh * np.max(mask)] = 1
        return mask

    def critical_region_estimation(self, features, output, class_idx, div, input_shape, bin_thresh=0.15):
        self.model.zero_grad()

        # 直接在gradcam里面改
        def CategoricalCrossEntropyLoss(y_hat, y):
            return torch.nn.NLLLoss()(torch.log(y_hat), y)

        prob = F.softmax(output / div, dim=-1)
        x = prob
        y = torch.tensor(class_idx).unsqueeze(dim=0).cuda()
        loss = -CategoricalCrossEntropyLoss(x, y)

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

    def negative_mask_generation(self, features, output, class_indices, input_shape, bin_thresh=0.15, wo_softmax=True):
        self.model.zero_grad()
        if wo_softmax:
            output[:, class_indices].backward(retain_graph=True)
        else:
            prob = F.softmax(output, dim=1)
            prob[:, class_indices].backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))

        final_mask = np.ones_like(input_shape)
        for mask_idx in len(class_indices):
            mask = np.zeros(target.shape[2:], dtype=np.float32)
            cur_weights = weights[mask_idx]
            for i, w in enumerate(cur_weights):
                mask += w * target[mask_idx, i, :, :]

            mask = self.mask_post_process(-1 * mask.copy(), bin_thresh, input_shape)  # negative mask
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
            filling_content = torch.randn_like(input[0].shape)

        surgical_removal = input.copy()
        surgical_removal[0, mask != 0] = filling_content[mask != 0]
        return surgical_removal

    def process(self, input, top_k, delta_r, filling_content=None, preprocess_input=None, sign=1, gen_negative=False,
                div=2):
        features_before, output_before = self.forward(input)

        original_idx = np.argmax(output_before.cpu().data.numpy())
        critical_region = self.critical_region_estimation(features=features_before, output=output_before,
                                                          class_idx=original_idx, div=div,
                                                          input_shape=input.shape[-3:-1])

        intermediate_input = self.content_filling(input, critical_region, filling_content)

        features_after, output_after = self.forward(intermediate_input)
        delta = output_after - output_before
        sorted_delta = np.argsort(delta[0].cpu().numpy())
        final_mask = self.negative_mask_generation(features=features_after, output=output_after,
                                                   class_indices=sorted_delta[-top_k:], input_shape=input.shape[-3:-1])
        final_input = self.content_filling(input, final_mask, filling_content)
        _, output_final = self.forward(final_input)
        ranking_delta = np.argsort(output_final[0].cpu().numpy()).index(original_idx)
        return ranking_delta < delta_r
