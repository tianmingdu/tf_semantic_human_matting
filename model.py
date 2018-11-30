# -*- coding: utf-8 -*-
"""
Created on Thu Nov  22 12:11:59 2018

@author: tianming
"""

import tensorflow as tf

from tensorflow.contrib.slim import nets

import preprocessing

slim = tf.contrib.slim


class Model(object):
    """xxx definition."""

    def __init__(self, is_training,
                 default_image_size=640,
                 alpha_loss_weight=10,
                 first_stage_image_loss_weight=0.5,
                 second_stage_alpha_loss_weight=0.5,
                 trimap_loss_weight=0.01):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
        """
        self._is_training = is_training
        self._default_image_size = default_image_size
        self._alpha_loss_weight = alpha_loss_weight
        self._first_stage_image_loss_weight = first_stage_image_loss_weight
        self._second_stage_alpha_loss_weight = second_stage_alpha_loss_weight
        self._trimap_loss_weight = trimap_loss_weight

    def preprocess(self, trimaps=None, images=None, images_forground=None,
                   images_background=None, alpha_mattes=None):
        """preprocessing.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            trimaps: A float32 tensor with shape [batch_size,
                height, width, 1] representing a batch of trimaps.
            images: A float32 tensor with shape [batch_size, height, width,
                3] representing a batch of images. Only passed values in case
                of test (i.e., in training case images=None).
            images_foreground: A float32 tensor with shape [batch_size,
                height, width, 3] representing a batch of foreground images.
            images_background: A float32 tensor with shape [batch_size,
                height, width, 3] representing a batch of background images.
            alpha_mattes: A float32 tensor with shape [batch_size,
                height, width, 1] representing a batch of groundtruth masks.
            
            
        Returns:
            The preprocessed tensors.
        """

        def _random_crop(t):
            num_channels = t.get_shape().as_list()[2]
            return preprocessing.random_crop_background(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size,
                channels=num_channels)

        def _border_expand_and_resize(t):
            return preprocessing.border_expand_and_resize(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size)

        def _border_expand_and_resize_g(t):
            return preprocessing.border_expand_and_resize(
                t, output_height=self._default_image_size,
                output_width=self._default_image_size,
                channels=1)

        preprocessed_images_fg = None
        preprocessed_images_bg = None
        preprocessed_alpha_mattes = None
        preprocessed_images = None
        preprocessed_trimaps = None

        if self._is_training:
            preprocessed_trimaps = tf.map_fn(_border_expand_and_resize_g, trimaps)
            preprocessed_trimaps = tf.to_float(preprocessed_trimaps)

            preprocessed_images_fg = tf.map_fn(_border_expand_and_resize,
                                               images_forground)
            preprocessed_alpha_mattes = tf.map_fn(_border_expand_and_resize_g,
                                                  alpha_mattes)
            images_background = tf.to_float(images_background)
            preprocessed_images_bg = tf.map_fn(_random_crop, images_background)

            preprocessed_images_fg = tf.to_float(preprocessed_images_fg)
            preprocessed_alpha_mattes = tf.to_float(preprocessed_alpha_mattes)
            preprocessed_images = (tf.multiply(preprocessed_alpha_mattes, preprocessed_images_fg) +
                                   tf.multiply(1 - preprocessed_alpha_mattes, preprocessed_images_bg))
        else:
            preprocessed_images_fg = tf.map_fn(_border_expand_and_resize, images_forground)
            preprocessed_images_fg = tf.to_float(preprocessed_images_fg)

        preprocessed_dict = {'images_fg': preprocessed_images_fg,
                             'images_bg': preprocessed_images_bg,
                             'alpha_mattes': preprocessed_alpha_mattes,
                             'images': preprocessed_images,
                             'trimaps': preprocessed_trimaps}
        return preprocessed_dict

    def pyramid_pooling(self, inputs, pool_size, depth, scope=None):
        with tf.variable_scope(scope, 'pyramid_pool_v1', [inputs]) as sc:
            dims = inputs.get_shape().dims
            out_height, out_width = dims[1].value, dims[2].value
            pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size, scope='pyramid_pool1')
            conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1, scope='pyramid_conv1')
            output = tf.image.resize_bilinear(conv1, [out_height, out_width])
            return output

    def predict(self, preprocessed_dict):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_dict: See The preprocess function.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_images_fg = preprocessed_dict.get('images_fg')

        net_image = slim.conv2d(preprocessed_images_fg, num_outputs=5, kernel_size=3,
                                padding='SAME', scope='psp_conv1')
        net_image = slim.batch_norm(net_image, is_training=self._is_training)
        net_image = slim.conv2d(net_image, num_outputs=4, kernel_size=3,
                                padding='SAME', scope='psp_conv2')
        net_image = slim.batch_norm(net_image, is_training=self._is_training)
        # pyramid scence pooling
        pool1 = self.pyramid_pooling(net_image, (60, 60), 64, scope='pyramid_pooling1')
        pool2 = self.pyramid_pooling(net_image, (30, 30), 64, scope='pyramid_pooling2')
        pool3 = self.pyramid_pooling(net_image, (20, 20), 64, scope='pyramid_pooling3')
        pool4 = self.pyramid_pooling(net_image, (10, 10), 64, scope='pyramid_pooling4')

        net_image = tf.concat(values=[net_image, pool1, pool2, pool3, pool4], axis=3)
        net_image = slim.batch_norm(net_image, is_training=self._is_training)
        pred_trimap = slim.conv2d(net_image, num_outputs=3, kernel_size=3,
                                  padding='SAME', scope='psp_conv3')
        pred_trimap_soft = tf.nn.softmax(pred_trimap, axis=3)
        # background_trimap = tf.slice(pred_trimap, [0, 0, 0, 0], [-1, -1, -1, 1])
        # foreground_trimap = tf.slice(pred_trimap, [0, 0, 0, 1], [-1, -1, -1, 1])
        # unsure_trimap = tf.slice(pred_trimap, [0, 0, 0, 2], [-1, -1, -1, 1])
        background = tf.slice(pred_trimap, [0, 0, 0, 0], [-1, -1, -1, 1])
        background_trimap = tf.slice(pred_trimap_soft, [0, 0, 0, 0], [-1, -1, -1, 1])
        foreground = tf.slice(pred_trimap, [0, 0, 0, 1], [-1, -1, -1, 1])
        foreground_trimap = tf.slice(pred_trimap_soft, [0, 0, 0, 1], [-1, -1, -1, 1])
        unsure = tf.slice(pred_trimap, [0, 0, 0, 2], [-1, -1, -1, 1])
        unsure_trimap = tf.slice(pred_trimap_soft, [0, 0, 0, 2], [-1, -1, -1, 1])

        # net_image_trimap = tf.concat(values=[preprocessed_images_fg, pred_trimap], axis=3)
        # VGG-16
        _, endpoints = nets.vgg.vgg_16(preprocessed_images_fg,
                                       num_classes=1,
                                       spatial_squeeze=False,
                                       is_training=self._is_training)
        # Note: The `padding` method of fc6 of VGG-16 in tf.contrib.slim is
        # `VALID`, but the expected value is `SAME`, so we must replace it.
        net_image = endpoints.get('vgg_16/pool5')
        net_image = slim.batch_norm(net_image, is_training=self._is_training)
        # net_image = slim.conv2d(net_image, num_outputs=4096, kernel_size=7,
        #                         padding='SAME', scope='fc6_')

        # VGG-16 for alpha channel
        net_alpha = slim.repeat(pred_trimap, 2, slim.conv2d, 64,
                                [3, 3], scope='conv1_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool1_alpha')
        net_alpha = slim.batch_norm(net_alpha, is_training=self._is_training)
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 128, [3, 3],
                                scope='conv2_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool2_alpha')
        net_alpha = slim.batch_norm(net_alpha, is_training=self._is_training)
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 256, [3, 3],
                                scope='conv3_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool3_alpha')
        net_alpha = slim.batch_norm(net_alpha, is_training=self._is_training)
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 512, [3, 3],
                                scope='conv4_alpha')
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool4_alpha')
        net_alpha = slim.repeat(net_alpha, 2, slim.conv2d, 512, [3, 3],
                                scope='conv5_alpha')
        net_alpha = slim.batch_norm(net_alpha, is_training=self._is_training)
        net_alpha = slim.max_pool2d(net_alpha, [2, 2], scope='pool5_alpha')
        # net_alpha = slim.conv2d(net_alpha, 4096, [7, 7], padding='SAME',
        #                         scope='fc6_alpha')
        net_alpha = slim.batch_norm(net_alpha, is_training=self._is_training)

        # Concate the first stage prediction
        net = tf.concat(values=[net_image, net_alpha], axis=3)
        net.set_shape([None, self._default_image_size // 32,
                       self._default_image_size // 32, 1024])

        # Deconvlution
        with slim.arg_scope([slim.conv2d_transpose], stride=2, kernel_size=5):
            # Deconv6
            net = slim.conv2d_transpose(net, num_outputs=512, kernel_size=1, scope='deconv6')
            net = slim.batch_norm(net, is_training=self._is_training)
            # Deconv5
            net = slim.conv2d_transpose(net, num_outputs=512, scope='deconv5')
            net = slim.batch_norm(net, is_training=self._is_training)
            # Deconv4
            net = slim.conv2d_transpose(net, num_outputs=256, scope='deconv4')
            net = slim.batch_norm(net, is_training=self._is_training)
            # Deconv3
            net = slim.conv2d_transpose(net, num_outputs=128, scope='deconv3')
            net = slim.batch_norm(net, is_training=self._is_training)
            # Deconv2
            net = slim.conv2d_transpose(net, num_outputs=64, scope='deconv2')
            net = slim.batch_norm(net, is_training=self._is_training)
            # Deconv1
            net = slim.conv2d_transpose(net, num_outputs=64, stride=1, scope='deconv1')
            net = slim.batch_norm(net, is_training=self._is_training)

        # Predict alpha matte
        alpha_matte_r = slim.conv2d(net, num_outputs=1, kernel_size=[5, 5],
                                    activation_fn=tf.nn.sigmoid,
                                    scope='AlphaMatte')

        alpha_matte_p = foreground_trimap + tf.multiply(unsure_trimap, alpha_matte_r)
        prediction_dict = {'alpha_matte_r': alpha_matte_r,
                           'alpha_matte_p': alpha_matte_p,
                           'pred_trimap': pred_trimap,
                           'background': background,
                           'foreground': foreground,
                           'background_trimap': background_trimap,
                           'foreground_trimap': foreground_trimap,
                           'unsure_trimap': unsure_trimap,
                           }
        return prediction_dict

    def postprocess(self, prediction_dict, preprocessed_dict=None, use_trimap=True):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        pred_images = None
        gt_images = None
        gt_fg = None
        gt_bg = None
        if (not preprocessed_dict is None):
            gt_fg = preprocessed_dict.get('images_fg')
            gt_bg = preprocessed_dict.get('images_bg')
            gt_images = preprocessed_dict.get('images')

        pred_trimap = prediction_dict.get('pred_trimap')
        alpha_matte_r = prediction_dict.get('alpha_matte_r')
        alpha_matte_p = prediction_dict.get('alpha_matte_p')

        threshold = tf.fill(tf.shape(alpha_matte_p), 0.9)
        bool_great = tf.math.greater(alpha_matte_p, threshold)
        zeros = tf.fill(tf.shape(alpha_matte_p), 0.0)
        ones = tf.fill(tf.shape(alpha_matte_p), 1.0)
        alpha_matte_p_discrete = tf.where(bool_great, ones, zeros)
        if (not preprocessed_dict is None):
            pred_images = tf.multiply(alpha_matte_p_discrete, gt_fg) + tf.multiply(1 - alpha_matte_p_discrete, gt_bg)

        postprocessed_dict = {'pred_trimap': (pred_trimap * 255.),
                              'alpha_matte_r': (alpha_matte_r * 255.),
                              'alpha_matte_p': (alpha_matte_p * 255.),
                              'alpha_matte_p_discrete': (alpha_matte_p_discrete * 255.),
                              'pred_image': pred_images,
                              'gt_images': gt_images}
        if (preprocessed_dict is None):
            postprocessed_dict = {'pred_trimap': (pred_trimap * 255.),
                                  'alpha_matte_r': (alpha_matte_r * 255.),
                                  'alpha_matte_p': (alpha_matte_p * 255.),
                                  'alpha_matte_p_discrete': (alpha_matte_p_discrete * 255.)
                                  }
        return postprocessed_dict

    def loss(self, prediction_dict, preprocessed_dict, epsilon=1e-12):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            preprocessed_dict: A dictionary of tensors holding groundtruth
                information, see preprocess function. The pixel values of 
                groundtruth_alpha_matte must be in [0, 128, 255].
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.
        """
        gt_images = preprocessed_dict.get('images')
        gt_fg = preprocessed_dict.get('images_fg')
        gt_bg = preprocessed_dict.get('images_bg')
        gt_alpha_matte = preprocessed_dict.get('alpha_mattes')
        gt_trimaps = preprocessed_dict.get('trimaps')

        pred_trimaps = prediction_dict.get('pred_trimap')
        alpha_matte_r = prediction_dict.get('alpha_matte_r')
        alpha_matte_p = prediction_dict.get('alpha_matte_p')
        background = prediction_dict.get('background')
        foreground = prediction_dict.get('foreground')

        pred_images = tf.multiply(alpha_matte_p, gt_fg) + tf.multiply(1 - alpha_matte_p, gt_bg)

        # weights = tf.where(tf.equal(pred_trimaps, 128),
        #                    tf.ones_like(pred_trimaps),
        #                    tf.zeros_like(pred_trimaps))
        # total_weights = tf.reduce_sum(weights) + epsilon

        # trimap_losses = tf.sqrt(tf.square(pred_trimaps - gt_trimaps) + epsilon)
        # trimap_losses = tf.reduce_mean(trimap_losses)

        trimap_losses = tf.losses.softmax_cross_entropy(
            tf.concat([gt_alpha_matte, 1 - gt_alpha_matte], axis=3),
            tf.concat([foreground, background], axis=3))
        alpha_losses = tf.sqrt(tf.square(alpha_matte_p - gt_alpha_matte) + epsilon)
        alpha_losses = tf.reduce_mean(alpha_losses)

        composition_losses = tf.sqrt(tf.square(pred_images / 255. - gt_images / 255.) + epsilon)
        composition_losses = tf.reduce_mean(composition_losses)

        loss = (self._alpha_loss_weight * alpha_losses +
                self._first_stage_image_loss_weight * composition_losses +
                self._trimap_loss_weight * trimap_losses)
        loss_dict = {'trimap_losses': trimap_losses,
                     'alpha_losses': alpha_losses,
                     'composition_losses': composition_losses,
                     'loss': loss}
        return loss_dict
