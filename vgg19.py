# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.
# Most code in this file was borrowed from https://github.com/anishathalye/neural-style/blob/master/vgg.py

import tensorflow as tf
import numpy as np
import scipy.io
import random

# download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
from tensorflow.python.ops.image_ops_impl import ResizeMethod

MODEL_FILE_NAME = 'imagenet-vgg-verydeep-19.mat'

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel

def undo_preprocess(image, mean_pixel):
    return image + mean_pixel

class VGG19:

    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self, data_path):
        data = scipy.io.loadmat(data_path)

        #self.mean_pixel = np.array([123.68, 116.779, 103.939])
        self.mean_pixel = np.array([0, 0, 0])

        self.weights = data['layers'][0]

    def preprocess(self, image):
        return image-self.mean_pixel

    def undo_preprocess(self,image):
        return image+self.mean_pixel

    def feed_forward(self, input_image, scope=None):
        net = {}
        current = input_image
        activations_all = []
        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]

                #IF NOT LAYER 6
                #if i!=0 and i!=1:
                #    net[name] = input_image
                #    if kind == 'relu':
                #        activations_all.append(input_image)
                #    continue

                if kind == 'conv':
                    kernels = self.weights[i][0][0][2][0][0]
                    bias = self.weights[i][0][0][2][0][1]

                    # THE FOLLOWING LINES IS FOR RANDOM KERNELS AND BIASES
                    #kernels = np.random.rand(kernels.shape[0], kernels.shape[1], kernels.shape[2], kernels.shape[3])
                    #kernels = np.float32(kernels)
                    #bias = np.random.rand(bias.shape[0], bias.shape[1])
                    #bias = np.float32(bias)

                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))

                    #THE FOLLOWING LINE TRUNCATES KERNELS SO THAT THEY WOULD OPERATE ON INPUT
                    #SO 3X3X64X128 BECOMES 3X3X3X128 BECAUSE IN_CHANNELS IS ALWAYS = 3
                    #kernels = kernels[:,:,0:3,:]

                    bias = bias.reshape(-1)

                    current = _conv_layer(current, kernels, bias)

                elif kind == 'relu':
                    current = tf.nn.relu(current)
                    #current = tf.nn.tanh(current)
                    #current = tf.nn.sigmoid(current)
                    activations_all.append(current)

                    #THE FOLLOWING LINE RESETS THE INPUT SO THAT THE NEXT LAYER APPLIES ON THE INPUT LAYER!
                    #current = input_image
                    activations = input_image

                    #if name == 'relu1_2':
                        #print shape of output of second convolutional layer (rows x cols x 64 channels)
                        #print_output = tf.Print(current, [tf.shape(current)[0], tf.shape(current)[1], tf.shape(current)[2], tf.shape(current)[3]], "#shape after 2nd conv: ")
                        #current = print_output
                        #activations = current
                        #print_output = tf.Print(current, [tf.shape(activations)[0], tf.shape(activations)[1], tf.shape(activations)[2], tf.shape(activations)[3]], "#shape after activations sum: ")
                        #current = print_output

                elif kind == 'pool':
                    current = _pool_layer(current)
                    print('do nothing')
                net[name] = current

        assert len(net) == len(self.layers)
        #return net, activations, activations_all
        return activations_all
