import tensorflow as tf

from modules.layers.tf_utils import get_runtime_coef, scaled_lrelu


class fully_connected(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 apply_bias=True, 
                 apply_lrelu=True, 
                 lr_mul=1.0,
                 **kwargs):

        super(fully_connected, self).__init__(**kwargs)
        self.units = units
        self.apply_bias = apply_bias
        self.apply_lrelu = apply_lrelu
        self.lr_mul = lr_mul

    def build(self, input_shape):
        # input_shape = [batch_size, input_units] or [batch_size, h, w, channels]
        weight_shape = [tf.reduce_prod(input_shape[1:]), self.units]
        init_std, self.runtime_coef = get_runtime_coef(self.lr_mul, weight_shape)    

        self.w = self.add_weight(name='fc_w',
                                 shape=weight_shape,
                                 dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, init_std),
                                 trainable=True)
        if self.apply_bias:
            self.b = self.add_weight(name='fc_b',
                                    shape=(self.units),
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer(),
                                    trainable=True)
                                    
    def call(self, inputs, training=None):
        # Equalized learning rate and custom learning rate multiplier.   
        w = self.w * self.runtime_coef
        
        x = tf.reshape(inputs, [tf.shape(inputs)[0], -1]) # Flatten
        x = tf.matmul(x, w)

        if self.apply_bias:
            b = self.b * self.lr_mul
            x = tf.nn.bias_add(x, b)
        if self.apply_lrelu:
            x = scaled_lrelu(alpha=0.2)(x)

        return x