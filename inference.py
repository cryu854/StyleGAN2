import tensorflow as tf
import numpy as np
import os

from PIL import Image
from utils import imsave, create_dir
from modules.generator import generator


class Inferencer:
    def __init__(self,
                 resolution,
                 num_labels,
                 config,
                 truncation_psi,
                 checkpoint_path,
                 result_path='./results',
                 **kwargs):

        self.resolution = resolution
        self.num_labels = num_labels
        self.truncation_psi = truncation_psi
        self.result_path = result_path
        self.Gs = generator(resolution, num_labels, config, randomize_noise=False)
        self.ckpt = tf.train.Checkpoint(generator_clone=self.Gs)
        print(f'Loading network from {checkpoint_path}...')
        self.ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
  

    def genetate_example(self, num_examples, batch_size=1, label=None):
        create_dir(f'{self.result_path}/example')
        print('Generating images...')
        for begin in range(0, num_examples, batch_size):
            latents = tf.random.normal([batch_size, 512])
            labels_indice = [label]*batch_size if label is not None else tf.random.uniform([batch_size], 0, self.num_labels, dtype=tf.int32)
            labels = tf.one_hot(labels_indice, self.num_labels) if self.num_labels > 0 else tf.zeros([batch_size, 0], tf.float32)
            images = self.Gs([latents, labels], self.truncation_psi, training=False)
            for idx, (image, label) in enumerate(zip(images, labels_indice)):
                imsave(image, f'{self.result_path}/example/{begin+idx}_label-{label}.jpg')


    def style_mixing_example(self, row_seeds, col_seeds, label=None, col_styles='0-6'):
        create_dir(f'{self.result_path}/mixing')
        all_seeds = list(set(row_seeds + col_seeds))
        all_labels_indice = [label]*len(all_seeds) if label is not None else tf.random.uniform([len(all_seeds)], 0, self.num_labels, dtype=tf.int32)
        all_labels = tf.one_hot(all_labels_indice, self.num_labels) if self.num_labels > 0 else tf.zeros([len(all_seeds), 0], tf.float32)
        all_z = tf.stack([tf.random.normal([512], seed=seed) for seed in all_seeds])    # [minibatch, component]

        print('Generating images...')
        all_images, all_w = self.Gs([all_z, all_labels], self.truncation_psi, return_latents=True, training=False)
        all_images = (all_images + 1) * 127.5
        all_images = np.clip(all_images, 0.0, 255.0).astype(np.uint8)
        w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]
        image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

        print('Generating style-mixed images...')
        for row_seed in row_seeds:
            for col_seed in col_seeds:
                w = w_dict[row_seed].numpy()
                for col_style in col_styles:
                    w[col_style] = w_dict[col_seed][col_style]
                image = self.Gs.synthesis(w[np.newaxis])[0]
                image = (image + 1) * 127.5
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
                image_dict[(row_seed, col_seed)] = image

        print('Saving images...')
        for (row_seed, col_seed), image in image_dict.items():
            Image.fromarray(image, 'RGB').save(f'{self.result_path}/mixing/{row_seed}-{col_seed}.jpg')

        print('Saving image grid...')
        canvas = Image.new('RGB', (self.resolution * (len(col_seeds) + 1), self.resolution * (len(row_seeds) + 1)), 'black')
        for row_idx, row_seed in enumerate([None] + row_seeds):
            for col_idx, col_seed in enumerate([None] + col_seeds):
                if row_seed is None and col_seed is None:
                    continue
                key = (row_seed, col_seed)
                if row_seed is None:
                    key = (col_seed, col_seed)
                if col_seed is None:
                    key = (row_seed, row_seed)
                canvas.paste(Image.fromarray(image_dict[key], 'RGB'), (self.resolution * col_idx, self.resolution * row_idx))
        canvas.save(f'{self.result_path}/mixing/grid.png')


    def generate_gif(self, output='test', label=None, num_rows=2, num_cols=3, resolution=256, num_phases=5, transition_frames=20, static_frames=5, seed=1000):
        create_dir(f'{self.result_path}/gif')
        output_seq = []
        batch_size = num_rows * num_cols
        latents = [tf.random.normal([batch_size, 512]) for _ in range(num_phases)]
        labels_indice = [label]*batch_size if label is not None else tf.random.uniform([batch_size], 0, self.num_labels, dtype=tf.int32)
        labels = [tf.one_hot(labels_indice, self.num_labels) if self.num_labels > 0 else tf.zeros([batch_size, 0], tf.float32) for _ in range(num_phases)]

        def to_image_grid(outputs):
            outputs = (outputs + 1) * 127.5
            outputs = np.clip(outputs, 0.0, 255.0).astype(np.uint8)
            outputs = np.reshape(outputs, [num_rows, num_cols, *outputs.shape[1:]])
            outputs = np.concatenate(outputs, axis=1)
            outputs = np.concatenate(outputs, axis=1)
            return Image.fromarray(outputs).resize((resolution * num_cols, resolution * num_rows), Image.ANTIALIAS)
            
        print('Generating images...')
        for i in range(num_phases):
            _, dlatents0 = self.Gs([latents[i - 1], labels[i - 1]], self.truncation_psi, return_latents=True, training=False)
            _, dlatents1 = self.Gs([latents[i], labels[i]], self.truncation_psi, return_latents=True, training=False)

            for j in range(transition_frames):
                dlatents = (dlatents0 * (transition_frames - j) + dlatents1 * j) / transition_frames
                output_seq.append(to_image_grid(self.Gs.synthesis(dlatents)))
            output_seq.extend([to_image_grid(self.Gs.synthesis(dlatents1))] * static_frames)
  
        output = f'{self.result_path}/gif/{output}.gif'
        output_seq[0].save(output, save_all=True, append_images=output_seq[1:], optimize=False, duration=50, loop=0)