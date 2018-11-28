# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:35:40 2018

@author: shirhe-lyh
"""

import cv2
import glob
import numpy as np
import os

import data_provider
import predictor


def border_expand(image, value=255, output_height=320, output_width=320):
    height, width = image.shape[:2]
    max_size = max(height, width)
    if len(image.shape) == 3:
        expanded_image = np.zeros((max_size, max_size, 3)) + value
    else:
        expanded_image = np.zeros((max_size, max_size)) + value
    if height > width:
        pad_left = (height - width) // 2
        expanded_image[:, pad_left:pad_left+width] = image
    else:
        pad_top = (width - height) // 2
        expanded_image[pad_top:pad_top+height] = image
    return expanded_image


if __name__ == '__main__':
    images_dir = './train_test'
    trimaps_dir = './datasets/trimaps'
    output_dir = './train_test_output'
    frozen_inference_graph_path = ('./training/frozen_inference_graph_pb/' +
                                   'frozen_inference_graph.pb')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    matting_predictor = predictor.Predictor(frozen_inference_graph_path,
                                            gpu_index='1')
    
    for image_path in glob.glob(os.path.join(images_dir, '*.*')):
        image = cv2.imread(image_path)
        image_name = image_path.split('/')[-1]
        trimap_path = os.path.join(trimaps_dir, image_name)
        trimap = cv2.imread(trimap_path, 0)
        trimap_b = data_provider.trimap(trimap)
        
        images = np.expand_dims(image, axis=0)
        trimaps = np.expand_dims(trimap_b, axis=0)
        
        alpha_mattes, refined_alpha_mattes = matting_predictor.predict(
            images, trimaps)
        
        alpha_matte = np.squeeze(alpha_mattes, axis=0)
        refined_alpha_matte = np.squeeze(refined_alpha_mattes, axis=0)
        alpha_matte = 255 * alpha_matte
        refined_alpha_matte = 255 * refined_alpha_matte
        
        image_path_prefix = image_name.split('.')[0]
        output_name = image_path_prefix + '_alpha.jpg'
        output_name_refined = image_path_prefix + '_refined_alpha.jpg'
        output_path = os.path.join(output_dir, output_name)
        output_path_refined = os.path.join(output_dir, output_name_refined)
        cv2.imwrite(output_path, alpha_matte.astype(np.uint8))
        cv2.imwrite(output_path_refined, refined_alpha_matte.astype(np.uint8))