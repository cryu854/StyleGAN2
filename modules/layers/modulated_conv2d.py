import tensorflow as tf

from modules.layers.fully_connected import fully_connected
from modules.layers.ops.upfirdn_2d import upsample_conv_2d
from modules.layers.tf_utils import get_runtime_coef


class modulated_conv2d(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 up=False,
                 apply_bias=True,
                 demodulate=True,
                 fused_modconv=True,
                 lr_mul=1.0,
                 impl='ref',
                 **kwargs):
        super(modulated_conv2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up = up
        self.apply_bias = apply_bias
        self.demodulate = demodulate
        self.fused_modconv = fused_modconv
        self.lr_mul = lr_mul
        self.impl = impl

    def build(self, input_shape):
        x_shape, w_latents_shape = input_shape  # x_shape = [batch_size, height, width, channels]
        weight_shape = [self.kernel_size, self.kernel_size, x_shape[-1], self.filters]
        init_std, self.runtime_coef = get_runtime_coef(self.lr_mul, weight_shape)    

        self.fully_connected = fully_connected(units=x_shape[-1], apply_lrelu=False, name='modulate')

        self.w = self.add_weight(name='w',
                                 shape=weight_shape,
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, init_std),
                                 trainable=True)
        if self.apply_bias:
            self.b = self.add_weight(name='b',
                                     shape=(self.filters),
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)

    def call(self, inputs, training=None):
        x, w_latents = inputs
        # Transform to channel first.
        x = tf.transpose(x, [0, 3, 1, 2])

        # Equalized learning rate and custom learning rate multiplier.
        w = self.w * self.runtime_coef
        ww = w[tf.newaxis] # [BkkIO] Introduce minibatch dimension.

        # Modulate.
        s = self.fully_connected(w_latents) + 1
        ww *= tf.cast(s[:, tf.newaxis, tf.newaxis, :, tf.newaxis], w.dtype) # [BkkIO] Scale input feature maps.

        # Demodulate.
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [BO] Scaling factor.
            ww *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, :] # [BkkIO] Scale output feature maps.

        # Reshape/scale input.
        if self.fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
            w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), [ww.shape[1], ww.shape[2], ww.shape[3], -1])
        else:
            x *= tf.cast(s[:, :, tf.newaxis, tf.newaxis], x.dtype) # [BIhw] Not fused => scale input activations.

        # Convolution with optional up/downsampling.
        if self.up:
            x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=[1,3,3,1], impl=self.impl)
        else:
            x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

        # Reshape/scale output.
        if self.fused_modconv:
            x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        elif self.demodulate:
            x *= tf.cast(d[:, :, tf.newaxis, tf.newaxis], x.dtype) # [BOhw] Not fused => scale output activations.
        
        # Transform back to channel last
        x = tf.transpose(x, [0, 2, 3, 1])

        if self.apply_bias:
            b = self.b * self.lr_mul
            x = tf.nn.bias_add(x, b)

        return x