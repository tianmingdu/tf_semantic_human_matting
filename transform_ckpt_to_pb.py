# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:02:08 2018

@author: shirhe-lyh
"""

import os
import tensorflow as tf

from tensorflow.python.framework import graph_util

flags = tf.flags

flags.DEFINE_string('checkpoint_path', None, 'Path to .ckpt file.')
flags.DEFINE_string('output_path', None, 'Saving path to .pb file.')

FLAGS = flags.FLAGS


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    with tf.Session() as sess:
        checkpoint_path = FLAGS.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = './training/model.ckpt-40000'
        output_path = FLAGS.output_path
        if output_path is None:
            output_path = './frozen/frozen_inference_graph.pb'
        
        # Load .ckpt file
        if checkpoint_path.endswith('.meta'):
            checkpoint_path_with_meta = checkpoint_path
            checkpoint_path = checkpoint_path.replace('.meta', '')
        else:
            checkpoint_path_with_meta = checkpoint_path + '.meta'
        if not tf.gfile.Exists(checkpoint_path_with_meta):
            raise ValueError('`checkpoint_path` does not exist.')
        saver = tf.train.import_meta_graph(checkpoint_path_with_meta)
        saver.restore(sess, checkpoint_path)
        
        # Save as .pb file
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            ['alpha_matte_r']
        )
        with tf.gfile.GFile(output_path, 'wb') as fid:
            serialized_graph = output_graph_def.SerializeToString()
            fid.write(serialized_graph)
            
            
if __name__ == '__main__':
    tf.app.run()
            
