import os
import tensorflow as tf

import keras.backend as K
import numpy as np
import cv2

from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt


class TaintRadarTF:
    def __init__(self, model, name_of_last_conv=None):
        self.__model = model
        self.__name_of_last_conv = name_of_last_conv
        self.__mask_iterate, self.__logits_iterate = self.__init_mask_iterate()
        self.__registered_image = []
        self.analyzer = NegativeMaskAnalyzer(model, name_of_last_conv)

    def __init_mask_iterate(self):
        logits = self.__model.output.op.inputs[0]
        div = tf.placeholder(dtype=tf.float32, shape=(1,))
        probs = K.softmax(logits / div)
        target_y = tf.placeholder(dtype=tf.float32, shape=(None, self.__model.output_shape[-1]))
        loss = -categorical_crossentropy(target_y, probs)
        if self.__name_of_last_conv:
            last_conv_layer = self.__model.get_layer(self.__name_of_last_conv)
        else:
            last_conv_layer = self.__model.get_layer('block5_conv3')

        grads_mask = K.gradients(loss, last_conv_layer.output)[0]
        grads_img = K.gradients(loss, self.__model.input)[0]
        pooled_grads = K.mean(grads_mask, axis=(0, 1, 2))
        mask_iterate = K.function([self.__model.input, target_y, div],
                                  [loss, pooled_grads, last_conv_layer.output[0], grads_img, probs])
        logits_iterate = K.function([self.__model.input], [logits])
        return mask_iterate, logits_iterate

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

        self.__clear_registered()

    def __clear_registered(self):
        self.__registered_image = []

    def get_grads(self, x, target=None, div=2, preprocess_input=None):
        if not preprocess_input:
            preprocess_input = lambda x: x
        preds = self.__model.predict(preprocess_input(x.copy()))
        if not target:
            target = np.argmax(preds[0])
        # print(target)
        targets = np.expand_dims(np.array([target]), axis=0)
        target_one_hot = to_categorical(targets, self.__model.output_shape[-1])

        loss, pooled_grads_value, conv_layer_output_value, grads_img, probs = self.__mask_iterate(
            [preprocess_input(x.copy()), target_one_hot, div])
        for i in range(pooled_grads_value.shape[-1]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (x.shape[1], x.shape[1]))

        heatmap = np.uint8(255 * heatmap)
        return heatmap, grads_img, target

    def get_logits(self, x):
        return self.__logits_iterate([x.copy()])[0]

    def critical_region_estimation(self, image, preprocess_input):
        mask, grads_img, original_idx = self.get_grads(preprocess_input(image.copy()))
        mask[mask < np.max(mask) * 0.15] = 0
        mask[mask >= np.max(mask) * 0.15] = 1
        return mask, original_idx

    def final_mask_generation(self, image, preprocess_input, class_indices):
        final_mask = np.ones((image.shape[0], image.shape[1]))
        sub_masks, _ = self.analyzer.generate_negative(preprocess_input(image.copy()), class_idx=class_indices,
                                                       wo_softmax=True)
        idx = 0
        for sub_mask in sub_masks:
            sub_mask = cv2.resize((sub_mask * 255).astype('uint8'), (image.shape[1], image.shape[1]))
            sub_mask[sub_mask < np.max(sub_mask) * 0.15] = 0
            sub_mask[sub_mask >= np.max(sub_mask) * 0.15] = 1
            final_mask = final_mask * sub_mask
            self.__register_fig((sub_mask * 255).astype('uint8'), title='sub_mask_{}'.format(class_indices[idx]))
            idx += 1
        final_mask = np.uint8(final_mask * 255)

        return final_mask

    def is_malicious(self, image, top_k, delta_r, filling_content=None, preprocess_input=None, verbose=False):
        self.__clear_registered()
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if not preprocess_input:
            preprocess_input = lambda x: x
        if filling_content is None:
            rng = np.random.RandomState(1)
            filling_content = rng.randint(0, 255, image[0].shape)
        self.__register_fig(image[0].astype('uint8'), title='original_input')

        mask, original_idx = self.critical_region_estimation(image, preprocess_input)

        self.__register_fig((mask * 255).astype('uint8'), title='estimation_mask')

        surgical_removal = image.copy()
        surgical_removal[0, mask != 0] = filling_content[mask != 0]

        self.__register_fig(surgical_removal[0].astype('uint8'), title='inter_input')

        act_before = np.array(self.get_logits(preprocess_input(image.copy())))[0]
        act_after = np.array(self.get_logits(preprocess_input(surgical_removal.copy())))[0]
        delta = act_after - act_before
        sorted_delta = np.argsort(delta)
        class_indices = sorted_delta[-top_k:]

        final_mask = self.final_mask_generation(image, preprocess_input, class_indices)

        self.__register_fig(final_mask, title='final_mask')
        final_input = image[0].copy()
        final_input[final_mask != 0] = filling_content[final_mask != 0]
        final_input = final_input.astype('uint8')
        self.__register_fig(final_input, title='final_input')
        final_input = np.expand_dims(final_input, axis=0)
        final_input = preprocess_input(final_input)
        preds = self.__model.predict(final_input)
        preds_sort = np.argsort(-preds[0])
        self.__plot_registered(verbose)
        return preds_sort.tolist().index(original_idx) > delta_r

    def save_registered(self, target_folder=None):
        im_num = len(self.__registered_image)
        for i in range(im_num):
            file_name = os.path.join(target_folder, self.__registered_image[i][1] + '.png')
            if len(self.__registered_image[i][0].shape) == 2:
                save_image = np.zeros((224, 224, 3))
                save_image[:, :, 0] = self.__registered_image[i][0]
                save_image[:, :, 1] = self.__registered_image[i][0]
                save_image[:, :, 2] = self.__registered_image[i][0]
                plt.imsave(file_name, save_image.astype('uint8'))
            else:
                plt.imsave(file_name, self.__registered_image[i][0].astype('uint8'))
        return self.__registered_image


MODEL_TO_LAST_CONV = {
    'inception_v3': 'conv2d_94',
    'vgg16': 'block5_conv3',
    'resnet50': 'res5c_branch2c',
    'vggface_vgg16': 'conv5_3',
}


class NegativeMaskAnalyzer:
    def __init__(self, model, name_of_last_conv='block5_conv3'):
        self.__model = model
        self.__name_of_last_conv = name_of_last_conv
        self.__iterate, self.__iterate_wo_softmax = self.__init_iterate(self.__model)

    def __init_iterate(self, model):
        last_conv_layer = model.get_layer(self.__name_of_last_conv)

        class_idx = tf.placeholder(dtype=tf.int32, shape=(None,))
        grads = tf.map_fn(lambda idx: K.gradients(model.output[:, idx], last_conv_layer.output)[0], class_idx,
                          tf.float32)

        grads_wo_softmax = tf.map_fn(
            lambda idx: K.gradients(model.output.op.inputs[0][:, idx], last_conv_layer.output)[0], class_idx,
            tf.float32)
        pooled_grads = K.mean(grads, axis=(1, 2, 3))
        pooled_grads_wo_softmax = K.mean(grads_wo_softmax, axis=(1, 2, 3))
        iterate = K.function([model.input, class_idx], [pooled_grads, last_conv_layer.output[0]])
        iterate_wo_softmax = K.function([model.input, class_idx], [pooled_grads_wo_softmax, last_conv_layer.output[0]])

        return iterate, iterate_wo_softmax

    def __get_output(self, x, class_idx=None, wo_softmax=True):
        if class_idx is None:
            preds = self.__model.predict(x)
            class_idx = np.argmax(preds[0])
        if type(class_idx) is int or type(class_idx) is np.int64:
            class_idx = np.expand_dims([class_idx], axis=0).reshape((1,))
        else:
            class_idx = np.array(class_idx).reshape((len(class_idx),))
        if wo_softmax:
            pooled_grads_value, conv_layer_output_value = self.__iterate_wo_softmax([x, class_idx])
        else:
            pooled_grads_value, conv_layer_output_value = self.__iterate([x, class_idx])
        return pooled_grads_value, conv_layer_output_value, class_idx

    def predict(self, x, wo_softmax=False):
        if wo_softmax:
            return self.__model_wo_softmax.predict(x)
        return self.__model.predict(x)

    def generate_negative(self, x, class_idx=None, wo_softmax=True, preprocess_input=None):
        if not preprocess_input:
            preprocess_input = lambda x: x
        pooled_grads_value, conv_layer_output_value, class_idx = self.__get_output(preprocess_input(x.copy()),
                                                                                   class_idx, wo_softmax)
        heatmaps = np.zeros((len(pooled_grads_value),
                             conv_layer_output_value.shape[0],
                             conv_layer_output_value.shape[1],
                             conv_layer_output_value.shape[2]))
        for i in range(pooled_grads_value.shape[-1]):
            for j in range(len(pooled_grads_value)):
                heatmaps[j, :, :, i] = conv_layer_output_value[:, :, i] * pooled_grads_value[j, i]

        heatmaps = np.mean(heatmaps, axis=-1)
        heatmaps = np.abs(np.minimum(heatmaps, 0))
        for i in range(len(heatmaps)):
            if np.max(heatmaps[i]) != 0:
                heatmaps[i] /= np.max(heatmaps[i])

        return heatmaps, class_idx[0]
