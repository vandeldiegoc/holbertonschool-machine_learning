#!/usr/bin/env python3
""" module """

import numpy as np
import tensorflow as tf


class NST:
    """ new class """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """ funtion """
        tf.enable_eager_execution()
        error1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        error2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(error1)
        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(error1)
        if not isinstance(content_image, np.ndarray):
            raise TypeError(error2)
        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(error2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.executing_eagerly()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """ Returns: the scaled image """
        error1 = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(error1)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(error1)
        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        h, w = image.shape[:-1]
        nw_shape_h = int(scale * h)
        nw_shape_w = int(scale * w)
        nw_shape = (nw_shape_h, nw_shape_w)
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, nw_shape)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        return image

    def load_model(self):
        """ saves the model in the instance attribute model """
        vgg = tf.keras.applications.VGG19(weights="imagenet",
                                          include_top=False,
                                          pooling='avg')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg.save('vgg_m')
        vgg_model = tf.keras.models.load_model('vgg_m',
                                               custom_objects=custom_objects)
        for layer in vgg_model.layers:
            layer.trainable = False

        style_outputs = [vgg_model.get_layer(name).output
                         for name in self.style_layers]
        content_outputs = vgg_model.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_outputs]
        self.model = tf.keras.models.Model(vgg_model.input,
                                           model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """ Returns: a tf.Tensor of shape (1, c, c)
            containing the gram matrix of input_layer """
        error3 = 'input_layer must be a tensor of rank 4'
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or len(input_layer.shape) != 4:
            raise TypeError(error3)

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = gram / tf.cast(n, tf.float32)
        gram = gram[tf.newaxis, :]
        return(gram)

    def generate_features(self):
        """ extracts the  style_features and content_feature"""
        vgg19 = tf.keras.applications.vgg19

        input_content_image = vgg19.preprocess_input(self.content_image * 255)
        input_style_image = vgg19.preprocess_input(self.style_image * 255)
        output_content_img = self.model(input_content_image)
        output_style_img = self.model(input_style_image)
        content_features = output_content_img[-1]
        style_features = []
        for output in output_style_img[:-1]:
            style_features = style_features + [self.gram_matrix(output)]
        self.gram_style_features = style_features
        self.content_feature = content_features

    def layer_style_cost(self, style_output, gram_target):
        """ Returns: the layer’s style cost """
        error4 = 'style_output must be a tensor of rank 4'
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(error4)

        c = int(style_output.shape[-1])
        err = 'gram_target must be a tensor of shape [1, {}, {}]'.format(c, c)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(err)
        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))
        return style_cost

    def style_cost(self, style_outputs):
        """calculate the style cost """
        err = "style_outputs must be a list with a length of {}".format(
            len(self.style_layers))
        if not isinstance(style_outputs, list):
            raise TypeError(err)
        if len(self.style_layers) != len(style_outputs):
            raise TypeError(err)

        style_costs = []
        weight = 1 / len(self.style_layers)

        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):

            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)

        style_cost = tf.add_n(style_costs)
        return style_cost
