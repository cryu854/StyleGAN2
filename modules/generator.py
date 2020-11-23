import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Lambda, Concatenate
from modules.layers.conv2d import conv2d
from modules.layers.modulated_conv2d import modulated_conv2d
from modules.layers.fully_connected import fully_connected
from modules.layers.tf_utils import scaled_lrelu
from modules.layers.ops.upfirdn_2d import upsample_2d


#----------------------------------------------------------------------------
# Layers for generator

class get_constant(Layer):
    def __init__(self, **kwargs):
        super(get_constant, self).__init__(**kwargs)
        self.constant = self.add_weight(name=self.name,
                                        shape=[1, 4, 4, 512],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(0, 1.0),
                                        trainable=True)
        
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        return tf.tile(self.constant, [batch_size, 1, 1, 1])


class label_embedding(Layer):
    def __init__(self, **kwargs):
        super(label_embedding, self).__init__(**kwargs)
        self.concate = Concatenate(axis=-1, name='concate')

    def build(self, input_shape):
        latents_shape, labels_shape = input_shape
        self.w = self.add_weight(name='labels_w',
                                 shape=[labels_shape[-1], 512],
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 1.0),
                                 trainable=True)

    def call(self, inputs, training=None):
        latents, labels = inputs
        labels = tf.matmul(labels, self.w)
        return self.concate([latents, labels])


class noise_injection(Layer):
    def __init__(self, randomize_noise, **kwargs):
        super(noise_injection, self).__init__(**kwargs)
        self.randomize_noise = randomize_noise

    def build(self, input_shape):
        self.noise_shape = input_shape
        self.noise = self.add_weight(name='noise',
                                     shape=[1, self.noise_shape[1], self.noise_shape[2], 1],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 1.0),
                                     trainable=False)
        self.noise_strength = self.add_weight(name='noise_strength',
                                              shape=[],
                                              dtype=tf.float32,
                                              initializer=tf.zeros_initializer(),
                                              trainable=True)

    def call(self, inputs, training=None):
        if self.randomize_noise:
            noise = tf.random.normal([tf.shape(inputs)[0], self.noise_shape[1], self.noise_shape[2], 1])
        else:
            noise = self.noise
        
        return inputs + noise * self.noise_strength


class gen_block(Layer):
    def __init__(self, filters, randomize_noise, up=False, impl='ref', **kwargs):
        super(gen_block, self).__init__(**kwargs)
        self.modulated_conv2d = modulated_conv2d(filters=filters, kernel_size=3, up=up, impl=impl)
        self.noise_injection = noise_injection(randomize_noise)
        self.scaled_lrelu = scaled_lrelu(alpha=0.2)

    def call(self, inputs, training=None):
        x, w_latents = inputs
        x = self.modulated_conv2d([x, w_latents])
        x = self.noise_injection(x)
        x = self.scaled_lrelu(x)
        return x


#----------------------------------------------------------------------------


class generator(Model):
    """ StyleGAN2's generator for config e/f with skip architecture."""
    def __init__(self,
                 resolution,
                 num_labels,
                 config,
                 impl='ref',
                 randomize_noise=True,
                 w_avg_beta=0.995,
                 style_mixing_prob=0.9,
                 **kwargs):
        super(generator, self).__init__(**kwargs)
        self.resolution = resolution
        self.res_log2 = int(np.log2(resolution))
        self.num_layers = tf.cast(self.res_log2 * 2 - 2, dtype=tf.int32)
        self.layer_idx = tf.range(self.num_layers)[tf.newaxis, :, tf.newaxis]
        self.mapping = self.g_mapping(num_labels)
        self.synthesis = self.g_synthesis(randomize_noise, config, impl)
        self.w_avg_beta = w_avg_beta
        self.style_mixing_prob = style_mixing_prob
        self.w_avg = tf.Variable(name='w_avg',
                                 initial_value=tf.zeros([512]),
                                 dtype=tf.float32,
                                 trainable=False)
        self.build([[None, 512], [None, num_labels]])


    def g_mapping(self, num_labels, name='g_mapping'):
        """ 8-layers mapping netrowk. """
        latents_in = Input(shape=(512,), name='latents_in')
        labels_in = Input(shape=(num_labels,), name='labels_in')
        z_latents = latents_in
        labesls = labels_in

        if num_labels > 0:
            z_latents = label_embedding(name='label_embedding')([z_latents, labesls])
        x = Lambda(lambda x: x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + 1e-8), name='pixel_norm')(z_latents)
        x = fully_connected(512, lr_mul=0.01, name='fc0')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc1')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc2')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc3')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc4')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc5')(x)
        x = fully_connected(512, lr_mul=0.01, name='fc6')(x)
        w_latents = fully_connected(512, lr_mul=0.01, name='fc7')(x)
        broadcasted_latents = Lambda(lambda x: tf.tile(x[:, tf.newaxis, :], [1, self.num_layers, 1]), name='broadcast_latents')(w_latents)

        latents_out = broadcasted_latents
        return Model(inputs=[latents_in, labels_in], outputs=[latents_out], name=name)


    def g_synthesis(self, randomize_noise, config, impl, name='g_synthesis'):
        """ Synthesis network with skip architecture. """
        filter_multiplier = 2 if config == 'f' else 1
        filters = {4: 512,
                   8: 512,
                   16: 512,
                   32: 512,
                   64: 256 * filter_multiplier,
                   128: 128 * filter_multiplier,
                   256: 64 * filter_multiplier,
                   512: 32 * filter_multiplier,
                   1024: 16 * filter_multiplier}

        latents_in = Input(shape=(self.num_layers, 512), name='latents_in')
        w_latents = latents_in

        constant = get_constant(name='constant')(w_latents)
        x = gen_block(filters=512, randomize_noise=randomize_noise, impl=impl, name='4x4')([constant, w_latents[:, 0]])
        y = modulated_conv2d(filters=3, kernel_size=1, demodulate=False, impl=impl, name='4x4_ToRGB')([x, w_latents[:, 1]])
        for index, (res, fmaps) in enumerate(list(filters.items())[1:self.res_log2-1]):
            x = gen_block(filters=fmaps, randomize_noise=randomize_noise, up=True, impl=impl, name=f'{res}x{res}_up')([x, w_latents[:, index*2+1]])
            x = gen_block(filters=fmaps, randomize_noise=randomize_noise, impl=impl, name=f'{res}x{res}')([x, w_latents[:, index*2+2]])
            y = Lambda(lambda x: upsample_2d(x, k=[1,3,3,1], data_format='NHWC', impl=impl), name=f'{res}x{res}_img_up')(y)
            y += modulated_conv2d(filters=3, kernel_size=1, demodulate=False, impl=impl, name=f'{res}x{res}_ToRGB')([x, w_latents[:, index*2+3]])
        
        images_out = y
        return Model(inputs=[latents_in], outputs=[images_out], name=name)


    def setup_as_moving_average_of(self, src_net, beta=0.99, beta_nontrainable=0.0):
        """ Updates the variables of this network to be slightly closer to those of the given network """
        for cur_weight, src_weight in zip(self.weights, src_net.weights):
            cur_beta = beta if cur_weight.trainable else beta_nontrainable
            new_weight = src_weight + (cur_weight-src_weight) * cur_beta    
            cur_weight.assign(new_weight)


    def update_moving_average(self, w_latents):
        """ Update moving average of W """
        batch_avg = tf.reduce_mean(w_latents[:, 0], axis=0)
        moved_w_avg = batch_avg + (self.w_avg - batch_avg) * self.w_avg_beta    
        self.w_avg.assign(moved_w_avg)
        return w_latents


    def style_mixing_regularization(self, latents_in, labels_in, w_latents):
        """ Perform style mixing regularization """
        latents2 = tf.random.normal(tf.shape(latents_in))
        w_latents2 = self.mapping([latents2, labels_in])
        if tf.random.uniform([], 0.0, 1.0) < self.style_mixing_prob:
            mixing_cutoff = tf.random.uniform([], 1, self.num_layers, dtype=tf.int32)
        else:
            mixing_cutoff = self.num_layers
        return tf.where(tf.broadcast_to(self.layer_idx < mixing_cutoff, tf.shape(w_latents)), w_latents, w_latents2)


    def truncation_trick(self, w_latents, truncation_psi):
        """ Apply truncation trick on W """
        layer_psi = tf.ones(self.layer_idx.shape, dtype=tf.float32)
        layer_psi *= truncation_psi
        truncated_w = self.w_avg + (w_latents - self.w_avg) * layer_psi  
        return truncated_w


    def call(self, inputs, truncation_psi=0.5, return_latents=False, training=None):
        latents_in, labels_in = inputs

        w_latents = self.mapping([latents_in, labels_in])
        if training:
            self.update_moving_average(w_latents)
            w_latents = self.style_mixing_regularization(latents_in, labels_in, w_latents)
        else:
            w_latents = self.truncation_trick(w_latents, truncation_psi)

        images_out = self.synthesis(w_latents)
        if return_latents:
            return images_out, w_latents
        return images_out