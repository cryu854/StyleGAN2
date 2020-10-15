import tensorflow as tf

from modules.layers.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
from modules.layers.tf_utils import get_runtime_coef, scaled_lrelu


class conv2d(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 up=False,
                 down=False, 
                 apply_bias=True, 
                 apply_lrelu=True, 
                 lr_mul=1.0,
                 impl='ref', 
                 **kwargs):
        super(conv2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.up = up
        self.down = down
        self.apply_bias = apply_bias
        self.apply_lrelu = apply_lrelu
        self.lr_mul = lr_mul
        self.impl = impl

    def build(self, input_shape):
        # input_shape = [batch_size, height, width, channels]
        weight_shape = [self.kernel_size, self.kernel_size, input_shape[-1], self.filters]
        init_std, self.runtime_coef = get_runtime_coef(self.lr_mul, weight_shape)    

        self.w = self.add_weight(name='conv_w',
                                 shape=weight_shape,
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, init_std),
                                 trainable=True)
        if self.apply_bias:
            self.b = self.add_weight(name='conv_b',
                                     shape=(self.filters),
                                     dtype=tf.float32,
                                     initializer=tf.zeros_initializer(),
                                     trainable=True)

    def call(self, inputs, training=None):
        # Equalized learning rate and custom learning rate multiplier.
        w = self.w * self.runtime_coef

        if self.up:
            x = upsample_conv_2d(inputs, w, data_format='NHWC', k=[1, 3, 3, 1], impl=self.impl)
        elif self.down:
            x = conv_downsample_2d(inputs, w, data_format='NHWC', k=[1, 3, 3, 1], impl=self.impl)
        else:
            x = tf.nn.conv2d(input=inputs, filters=w, data_format='NHWC', strides=[1, 1, 1, 1], padding='SAME')
        
        if self.apply_bias:
            b = self.b * self.lr_mul
            x = tf.nn.bias_add(x, b)
        if self.apply_lrelu:
            x = scaled_lrelu(alpha=0.2)(x)

        return x