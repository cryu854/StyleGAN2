import os
import time
import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D


class FID:
    def __init__(self, num_images, num_labels, batch_size, **kwargs):
        self.num_images = num_images
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', pooling='avg') 

    def _create_dataset(self, dataset_path):
        def parse_file(file_name):
            image = tf.io.read_file(file_name)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
            image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
            image = tf.keras.applications.inception_v3.preprocess_input(image * 255.0)
            return image
        dataset = tf.data.Dataset.list_files([dataset_path+'/*.png',dataset_path+'/*.jpg'], shuffle=True)
        dataset = dataset.map(parse_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def _evaluate_fakes_step(self, Gs):
        latents = tf.random.normal([self.batch_size, 512])
        labels_indice = tf.random.uniform([self.batch_size], 0, self.num_labels, dtype=tf.int32)
        labels = tf.one_hot(labels_indice, self.num_labels) if self.num_labels > 0 else tf.zeros([self.batch_size, 0], tf.float32)
        images = Gs([latents, labels], truncation_psi=1.0, training=False)
        images = tf.image.resize(images, [299, 299], method=tf.image.ResizeMethod.BILINEAR)
        images = tf.clip_by_value(images, clip_value_min=-1, clip_value_max=1)
        feats = self.inception_v3(images)
        return feats

    def evaluate(self, Gs, real_dir=None):
        cache_file = f'{real_dir}/FID-{self.num_images}-cache.npy'

        # Calculate mean and covariance statistics for reals.
        if os.path.isfile(cache_file):
            print(f'Restore real statistics from {cache_file}')
            with open(cache_file, 'rb') as f:
                real_mu = np.load(f)
                real_sigma = np.load(f)
        else:
            print('Start evaluating real statistics...')
            real_dataset = self._create_dataset(real_dir)
            feats = []
            for reals in real_dataset.take(self.num_images//self.batch_size):
                feats.append(self.inception_v3(reals).numpy())
            feats = np.concatenate(feats, axis=0)
            real_mu = np.mean(feats, axis=0)
            real_sigma = np.cov(feats, rowvar=False)
            with open(cache_file, 'wb') as f:
                np.save(f, real_mu)
                np.save(f, real_sigma)

        # Calculate mean and covariance statistics for fakes.
        print('Start evaluating fake statistics...')
        feats = []
        for _ in range(0, self.num_images, self.batch_size):
            feats.append(self._evaluate_fakes_step(Gs).numpy())
        feats = np.concatenate(feats, axis=0)
        fake_mu = np.mean(feats, axis=0)
        fake_sigma = np.cov(feats, rowvar=False)

        real_mu = np.atleast_1d(real_mu)
        fake_mu = np.atleast_1d(fake_mu)
        real_sigma = np.atleast_2d(real_sigma)
        fake_sigma = np.atleast_2d(fake_sigma)
        # calculate FID.
        ssdiff = np.sum(np.square(fake_mu - real_mu)) 
        covmean, _ = sqrtm(np.dot(fake_sigma, real_sigma), disp=False)
        dist = ssdiff + np.trace(fake_sigma + real_sigma - 2.0 * covmean.real)
        return dist


class PPL:
    def __init__(self, num_images, num_labels, epsilon, space, sampling, crop, batch_size, **kwargs):
        self.num_images = num_images
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.space = space
        self.sampling = sampling
        self.crop = crop
        self.batch_size = batch_size
        self.lpips = LPIPS()    # Learned perceptual metric.
        weights_path = tf.keras.utils.get_file(
                        'lpips.h5',
                        'https://drive.google.com/u/2/uc?id=12qqzTA2HGM50Qnt12FYyjgL6p4nl5bNZ&export=download',
                         cache_subdir='weights-lpips')   # Store at ~/.keras/weights-lpips
        self.lpips.load_weights(filepath=weights_path)

    def _normalize(self, v):
        """ Normalize batch of vectors. """
        return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

    def _lerp(self, a, b, t):
        """ Linear interpolation of a batch of vectors. """
        return a + (b - a) * t

    def _slerp(self, a, b, t):
        """ Spherical interpolation of a batch of vectors. """
        a = self._normalize(a)
        b = self._normalize(b)
        d = tf.reduce_sum(a * b, axis=-1, keepdims=True)
        p = t * tf.math.acos(d)
        c = self._normalize(b - d * a)
        d = a * tf.math.cos(p) + c * tf.math.sin(p)
        return self._normalize(d)

    @tf.function
    def _evaluate_step(self, Gs):
        # Generate random latents and interpolation t-values.
        lat_t01 = tf.random.normal([self.batch_size * 2, 512])
        lerp_t = tf.random.uniform([self.batch_size], 0.0, 1.0 if self.sampling == 'full' else 0.0)
        labels = tf.one_hot(tf.random.uniform([self.batch_size*2], 0, self.num_labels, dtype=tf.int32), self.num_labels) if self.num_labels > 0 else tf.zeros([self.batch_size*2, 0])

        if self.space == 'w':
            dlat_t01 = Gs.mapping([lat_t01, labels])
            dlat_t0, dlat_t1 = dlat_t01[0::2], dlat_t01[1::2]
            dlat_e0 = self._lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis])
            dlat_e1 = self._lerp(dlat_t0, dlat_t1, lerp_t[:, tf.newaxis, tf.newaxis] + self.epsilon)
            dlat_e01 = tf.reshape(tf.stack([dlat_e0, dlat_e1], axis=1), dlat_t01.shape)
        else:   # self.space == 'z'
            lat_t0, lat_t1 = lat_t01[0::2], lat_t01[1::2]
            lat_e0 = self._slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis])
            lat_e1 = self._slerp(lat_t0, lat_t1, lerp_t[:, tf.newaxis] + self.epsilon)
            lat_e01 = tf.reshape(tf.stack([lat_e0, lat_e1], axis=1), lat_t01.shape)
            dlat_e01 = Gs.mapping([lat_e01, labels])

        # Synthesize images.
        images = Gs.synthesis(dlat_e01)

        # Crop only the face region.
        if self.crop:
            c = int(images.shape[2] // 8)
            images = images[:, :, c*3 : c*7, c*2 : c*6]

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        factor = images.shape[2] // 256
        if factor > 1:
            images = tf.reshape(images, [-1, images.shape[1] // factor, factor, images.shape[2] // factor, factor, images.shape[-1]])
            images = tf.reduce_mean(images, axis=[2,4])

        # Evaluate perceptual distance.
        img_e0, img_e1 = images[0::2], images[1::2]
        dist = self.lpips([img_e0, img_e1]) * (1 / self.epsilon**2)
        return dist

    def evaluate(self, Gs):
        # Sampling loop.
        all_distances = []
        print('Start evaluating PPL...')
        for _ in range(0, self.num_images, self.batch_size):
            dist = self._evaluate_step(Gs).numpy()
            all_distances.append(dist)
        all_distances = np.concatenate(all_distances, axis=0)

        # Reject outliers.
        lo = np.percentile(all_distances, 1, interpolation='lower')
        hi = np.percentile(all_distances, 99, interpolation='higher')
        filtered_distances = np.extract(np.logical_and(lo <= all_distances, all_distances <= hi), all_distances)
        dist = np.mean(filtered_distances)
        return dist

# Learned perceptual metric
class LPIPS(Model):
    def __init__(self, lpips=True, spatial=False):
        super(LPIPS, self).__init__()
        self.lpips = lpips
        self.spatial = spatial
        self.vgg16 = self._vgg16_layers()
        self.lin_layers = [Conv2D(filters=1, kernel_size=1, strides=1, use_bias=False, name=f'linear_{idx}') for idx in range(5)]
        self.L = len(self.lin_layers)
        self.shift = tf.constant([-.030,-.088,-.188])[tf.newaxis, tf.newaxis, tf.newaxis, :]
        self.scale = tf.constant([.458,.448,.450])[tf.newaxis, tf.newaxis, tf.newaxis, :]
        self.build([[None, None, None, 3], [None, None, None, 3]])

    def _vgg16_layers(self):
        layer_names = ['block1_conv2',
                       'block2_conv2',
                       'block3_conv3', 
                       'block4_conv3', 
                       'block5_conv3']
        vgg16 = tf.keras.applications.VGG16(include_top=False, weights=None)
        outputs = [vgg16.get_layer(name).output for name in layer_names]
        return Model([vgg16.input], outputs)

    def _normalize_tensor(self, inputs, epsilon=1e-10):
        norm_factor = tf.math.sqrt(tf.reduce_sum(inputs**2, axis=-1, keepdims=True))
        return inputs / (norm_factor + epsilon)

    def _spatial_average(self, inputs, keepdims=True):
        return tf.reduce_mean(inputs, axis=[1,2], keepdims=keepdims)

    def _upsample(self, inputs, out_HW=(64,64)): # assumes scale factor is same for H and W
        return UpSampling2D(size=out_HW, mode='bilinear')(inputs)

    def call(self, inputs, training=None):
        """ Expected inputs range between [1, -1] """
        imgs1, imgs2 = inputs

        preprocessed_imgs1 = (imgs1 - self.shift) / self.scale
        preprocessed_imgs2 = (imgs2 - self.shift) / self.scale
        outs1 = self.vgg16(preprocessed_imgs1)
        outs2 = self.vgg16(preprocessed_imgs2)

        diffs = []
        for out1, out2 in zip(outs1, outs2):
            feats1, feats2 = self._normalize_tensor(out1), self._normalize_tensor(out2)
            diffs.append((feats1-feats2)**2)

        if(self.lpips):
            if(self.spatial):
                res = [self._upsample(self.lin_layers[kk](diffs[kk]), out_HW=imgs1.shape[1:-1]) for kk in range(self.L)]
            else:
                res = [self._spatial_average(self.lin_layers[kk](diffs[kk]), keepdims=True) for kk in range(self.L)]
        else:
            if(self.spatial):
                res = [self._upsample(diffs[kk].sum(dim=1,keepdims=True), out_HW=imgs1.shape[1:-1]) for kk in range(self.L)]
            else:
                res = [self._spatial_average(diffs[kk].sum(dim=1,keepdims=True), keepdims=True) for kk in range(self.L)]

        val = sum(res)
        return val