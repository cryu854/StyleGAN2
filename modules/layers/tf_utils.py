import tensorflow as tf


def get_runtime_coef(lr_mul, shape, gain=1.0, use_wscale=True):
    fan_in = tf.reduce_prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out]
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.math.sqrt(fan_in) # He init
 
    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lr_mul
        runtime_coef = he_std * lr_mul
    else:
        init_std = he_std / lr_mul
        runtime_coef = lr_mul

    return init_std, runtime_coef


class scaled_lrelu(tf.keras.layers.Layer):
    def __init__(self,
                 alpha=0.2,
                 **kwargs):

        super(scaled_lrelu, self).__init__(**kwargs)
        self.alpha = alpha
        self.scale = tf.math.sqrt(2.0)

    def call(self, inputs, training=None):
        return tf.nn.leaky_relu(inputs, alpha=self.alpha) * self.scale
