# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 16:25:17 2018

@author: tianming
"""

import cv2
import glob
import numpy as np
import os

import data_provider
import predictor
import shutil


def border_expand(image, value=255, output_height=320, output_width=320):
    height, width = image.shape[:2]
    max_size = max(height, width)
    if len(image.shape) == 3:
        expanded_image = np.zeros((max_size, max_size, 3)) + value
    else:
        expanded_image = np.zeros((max_size, max_size)) + value
    if height > width:
        pad_left = (height - width) // 2
        expanded_image[:, pad_left:pad_left + width] = image
    else:
        pad_top = (width - height) // 2
        expanded_image[pad_top:pad_top + height] = image
    return expanded_image


if __name__ == '__main__':
    # images_dir = '/data2/raycloud/deep_image_matting/test_images'
    images_dir = '/data1/shape_mask/datasets/images'
    frozen_inference_graph_path = ('./frozon/' +
                                   'frozen_inference_graph.pb')

    matting_predictor = predictor.Predictor(frozen_inference_graph_path,
                                            gpu_index='1')
    for image_path in glob.glob(os.path.join(images_dir, '*.*')):
        inputFilepath = image_path
        filename_w_ext = os.path.basename(inputFilepath)
        filename, file_extension = os.path.splitext(filename_w_ext)
        shutil.copyfile(image_path, './pred_out/' + filename + file_extension)
        image = cv2.imread(image_path)
        # alpha = np.zeros(image.shape[:2])
        # trimap = data_provider.trimap(alpha)

        # image = border_expand(image)
        # trimap = border_expand(trimap, value=0)

        images = np.expand_dims(image, axis=0)
        # trimaps = np.expand_dims(trimap, axis=0)

        alpha_matte_p, alpha_matte_p_discrete = matting_predictor.predict(
            images)

        alpha_matte_p = np.squeeze(alpha_matte_p, axis=0)
        alpha_matte_p_discrete = np.squeeze(alpha_matte_p_discrete, axis=0)
        alpha_matte_p =  alpha_matte_p
        alpha_matte_p_discrete =  alpha_matte_p_discrete

        image_path_prefix = './pred_out/' + filename
        output_path = image_path_prefix + '_alpha.jpg'
        output_path_refined = image_path_prefix + '_refined_alpha.jpg'
        cv2.imwrite(output_path, alpha_matte_p.astype(np.uint8))
        cv2.imwrite(output_path_refined, alpha_matte_p_discrete.astype(np.uint8))
