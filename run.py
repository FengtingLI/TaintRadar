import os
import numpy as np
from utils_tf import load_images
from tqdm import tqdm
from TaintRadarTF import TaintRadarTF
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

if __name__ == "__main__":
    model_path = os.path.join('models', 'vgg16.h5')
    model = VGG16(include_top=True, weights=model_path)
    taint_radar = TaintRadarTF(model, 'block5_conv3')

    image_folder = "images"

    attacked_list = np.array([path for path in os.listdir(image_folder) if not 'ori' in path and '.png' in path])
    original_list = np.array([path.split('.')[0] + '_origin.png' for path in attacked_list])

    attacked_images = load_images(image_folder, attacked_list, shape=(224, 224))
    original_images = load_images(image_folder, original_list, shape=(224, 224))

    result = []
    attacked_result = [taint_radar.is_malicious(attacked_images[i], top_k=9, delta_r=2,
                                           preprocess_input=preprocess_input, verbose=False)
                       for i in tqdm(range(len(attacked_images)))]
    original_result = [taint_radar.is_malicious(original_images[i], top_k=9, delta_r=2,
                                           preprocess_input=preprocess_input, verbose=False)
                       for i in tqdm(range(len(original_images)))]

    print("TPR: {}, FPR: {}".format(sum(attacked_result) / len(attacked_result),
                                    sum(original_result) / len(original_result)))
