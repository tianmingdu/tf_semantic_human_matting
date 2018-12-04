# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 11:19:50 2018

@author: tianming
"""

import cv2
import os
import data_provider
import glob
import numpy as np
from random import shuffle

def corresponding(image_png_files, image_bg_files, to_N=1):
    index = 0
    np.random.shuffle(image_png_files)
    np.random.shuffle(image_bg_files)
    num_bg_files = len(image_bg_files)
    correspondence_results = []
    i=0
    for image_file in image_png_files:
        for i in range(index, index+to_N):
            image_bg_file = image_bg_files[i%num_bg_files]
            correspondence_results.append([image_file, image_bg_file, str(i)+'.jpg'])
            i+=1
            index += to_N
    return correspondence_results



if __name__ == '__main__':
    images_fg_dir = './fg_images'
    images_bg_dir = './bg_images'
    trimap_dir = './gt_trimaps'
    alpha_dir = './gt_alphas'
    alpha_dis_dir = './gt_alpha_dis'
    corresponing_dir = './'

    if not os.path.exists(trimap_dir):
        os.mkdir(trimap_dir)
    if not os.path.exists(alpha_dir):
        os.mkdir(alpha_dir)
    if not os.path.exists(alpha_dis_dir):
        os.mkdir(alpha_dis_dir)
    if not os.path.exists(corresponing_dir):
        os.mkdir(corresponing_dir)

    images_fg_files = glob.glob(os.path.join(images_fg_dir, '*.png'))
    image_bg_files = glob.glob(os.path.join(images_bg_dir, '*.jpg'))
    print(image_bg_files)
    correspondence_results= corresponding(images_fg_files, image_bg_files, 3)
    with open(corresponing_dir+"correspondence.txt", 'w') as writer:
        for image_png_path, image_bg_path, new_name in correspondence_results:
            writer.write(image_png_path + '@' + image_bg_path + '@'+ os.path.join(trimap_dir,new_name)+ '\n')
            image = cv2.imread(image_png_path, -1)
            alpha = data_provider.alpha_matte(image)
            alpha_dis = data_provider.alpha_matte_discrete(image)
            trimap_gt = data_provider.trimap(alpha_dis, kernel_size_low=10, 
                                            kernel_size_high=60)
            cv2.imwrite(os.path.join(trimap_dir,new_name), trimap_gt)
            cv2.imwrite(os.path.join(alpha_dir,new_name), alpha)
            cv2.imwrite(os.path.join(alpha_dis_dir,new_name), alpha_dis)
    
    # for image_fg_name, image_bg_name in correspondence_results:
    #     output_path = os.path.join(output_dir, trimap_path)
    #     if os.path.exists(output_path):
    #         continue
        


