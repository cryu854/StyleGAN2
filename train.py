import tensorflow as tf
import numpy as np
import time
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from utils import imsave
from modules.generator import generator
from modules.discriminator import discriminator
from modules.losses import ns_pathreg_r1, ns_DiffAugment_r1


class Trainer:
    def __init__(self,
                 resolution=1024,
                 config='f',
                 batch_size=4,
                 total_img=25000000,
                 dataset_name='afhq',
                 dataset_path='./../../datasets/afhq/train_labels',
                 checkpoint_path='./checkpoint',
                 impl='ref',
                 **kwargs):
        
        """ Training parameters. """
        self.max_steps = total_img // batch_size
        self.resolution = resolution
        self.batch_size = batch_size
        self.impl = impl
        self.config = config
        self.dataset_name = dataset_name
        self.G_reg_interval = 8
        self.D_reg_interval = 16
        G_mb_ratio = self.G_reg_interval / (self.G_reg_interval + 1)
        D_mb_ratio = self.D_reg_interval / (self.D_reg_interval + 1)
    
        """ Training objects. """
        self.train_dataset, self.num_labels = self.create_dataset(dataset_path)
        self.D, self.G, self.Gs = self.build_model()
        self.loss_func, self.pl_mean = self.create_loss_func()
        self.G_opt = Adam(learning_rate=0.0025*G_mb_ratio, beta_1=0.0**G_mb_ratio, beta_2=0.99**G_mb_ratio, epsilon=1e-8)
        self.D_opt = Adam(learning_rate=0.0025*D_mb_ratio, beta_1=0.0**D_mb_ratio, beta_2=0.99**D_mb_ratio, epsilon=1e-8)

        """ Misc """
        self.step = tf.Variable(name='step', initial_value=0, trainable=False, dtype=tf.int32)
        self.elapsed_time = tf.Variable(name='elapsed_time', initial_value=0, trainable=False, dtype=tf.int32)
        self.create_summary()
        self.print_step = 100
        self.save_step = 1000
        self.ckpt = tf.train.Checkpoint(step=self.step,
                                        elapsed_time=self.elapsed_time,
                                        generator=self.G,
                                        discriminator=self.D,
                                        generator_clone=self.Gs,
                                        pl_mean=self.pl_mean)
        self.ckpt_manager = tf.train.CheckpointManager(checkpoint=self.ckpt, 
                                                       directory=checkpoint_path,
                                                       max_to_keep=2)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print(f"Restored from {self.ckpt_manager.latest_checkpoint} at step {self.ckpt.step.numpy()}.")
            if self.step.numpy() >= self.max_steps:
                print("Training has already completed.")
                return
        else:
            print("Initializing from scratch...")


    def build_model(self):
        """ Build initial model """
        D = discriminator(self.resolution, self.num_labels, self.config, self.impl)
        G = generator(self.resolution, self.num_labels, self.config, self.impl)
        Gs = generator(self.resolution, self.num_labels, self.config, self.impl)
        # Setup Gs's weights same as G
        Gs.set_weights(G.get_weights())
        print('G_trainable_parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in G.trainable_variables]))
        print('D_trainable_parameters:', np.sum([np.prod(v.get_shape().as_list()) for v in D.trainable_variables]))
        return D, G, Gs


    def create_dataset(self, dataset_path):
        """ Create dataset with one of 'ffhq'/'afhq'/'custom'."""
        @tf.function
        def parse_file(file_name):
            image = tf.io.read_file(file_name)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize(image, [self.resolution, self.resolution], method=tf.image.ResizeMethod.BILINEAR)
            label_name = tf.strings.split(file_name, sep='_')[-2]
            label = tf.math.argmax(label_name == class_names, output_type=tf.int32) if tf.size(class_names) > 0 else 0
            return image * 2.0 - 1.0, tf.one_hot(label, depth=tf.size(class_names))
            
        print(f'Creating {self.dataset_name} dataset...')
        if self.dataset_name == 'ffhq':
            # FFHQ dataset with no label from the paper
            # "Analyzing and Improving the Image Quality of StyleGAN" In CVPR 2020:
            # https://github.com/NVlabs/stylegan2
            class_names = []
        elif self.dataset_name == 'afhq':
            # AFHQ dataset with 3 labels(cat, dog, wild) from the paper
            # "StarGAN v2: Diverse Image Synthesis for Multiple Domains" In CVPR 2020:
            # https://github.com/clovaai/stargan-v2
            class_names = ['cat', 'dog', 'wild']
        else:   
            # Custom dataset will use DiffAugemnt to train, DiffAugment from the paper
            # "Differentiable Augmentation for Data-Efficient GAN Training" In NeurIPS 2020:
            # https://github.com/mit-han-lab/data-efficient-gans
            class_names = []

        dataset = tf.data.Dataset.list_files(dataset_path+'/*.jpg')
        dataset = dataset.shuffle(buffer_size=len(list(dataset)))
        dataset = dataset.map(parse_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset, tf.size(class_names)


    def create_loss_func(self):
        """ Create loss function and initial pl_mean value. """
        pl_mean = tf.Variable(name='pl_mean', initial_value=0.0, trainable=False, dtype=tf.float32)
        
        if self.dataset_name in ['ffhq', 'afhq']:
            """ Use StyleGAN2 final loss function. (Non-saturation logistic loss + r1 reg + pathlength reg) """
            loss_func = ns_pathreg_r1(self.G, self.D, self.batch_size, self.num_labels, pl_mean=pl_mean)
        else:
            """ Use DiffAugment to expand custom dataset. (Non-saturation logistic loss + r1 reg) """
            loss_func = ns_DiffAugment_r1(self.G, self.D, self.batch_size, policy='color,translation,cutout')
        return loss_func, pl_mean


    def train(self):
        """ Call this func to train StyleGAN2. """
        print(f'{self.max_steps} training steps for resolution {self.resolution}x{self.resolution}.')
        print(f'Current step is {self.step.numpy()}.')
        print('Start training...')
        start = time.perf_counter()

        for real_images, real_labels in self.train_dataset:

            metrics = self.train_step(real_images, real_labels)

            cur_step = self.step.numpy()

            if cur_step % self.print_step == 0:
                """ Write losses into summary and print training info. """
                self.elapsed_time.assign_add(round(time.perf_counter()-start))
                start = time.perf_counter()
                self.update_summary(metrics, cur_step, self.elapsed_time.numpy())

            if cur_step % self.save_step == 0:
                """ Save ckpt and generate validation results """
                self.ckpt_manager.save(checkpoint_number=self.step)

                latents = tf.random.normal([1, 512])
                labels = tf.one_hot(tf.random.uniform([1], 0, self.num_labels, dtype=tf.int32), self.num_labels) if self.num_labels > 0 else tf.zeros([1, 0])
                images = self.Gs([latents, labels], truncation_psi=0.5, training=False)
                imsave(images[0], f'./validate_{cur_step}.jpg')

            if cur_step >= self.max_steps:
                break
  
        print(f'Total Time taken is {self.elapsed_time.numpy()} sec\n')
        self.ckpt_manager.save(checkpoint_number=self.step)


    @tf.function
    def train_step(self, real_images, real_labels):
        """ Update D, G and setup Gs weights per train step. """
        self.step.assign_add(1)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            """ Discriminator training step. """
            tape.watch(self.D.trainable_variables)

            if self.step % self.D_reg_interval == 0:
                """ With r1 regulation. """
                D_loss, D_reg = self.loss_func.get_D_loss(real_images, real_labels, compute_reg=True)
                D_loss += tf.reduce_mean(D_reg * self.D_reg_interval)
            else:
                D_loss, _ = self.loss_func.get_D_loss(real_images, real_labels)

        D_loss_grads = tape.gradient(D_loss, self.D.trainable_variables)
        self.D_opt.apply_gradients(zip(D_loss_grads, self.D.trainable_variables))


        with tf.GradientTape(watch_accessed_variables=False) as tape:
            """ Generator training step. """
            tape.watch(self.G.trainable_variables)

            if self.step % self.G_reg_interval == 0: 
                """ With pl regulation. """
                G_loss, G_reg = self.loss_func.get_G_loss(real_images, real_labels, compute_reg=True)
                G_loss += tf.reduce_mean(G_reg * self.G_reg_interval)
            else:
                G_loss, _ = self.loss_func.get_G_loss(real_images, real_labels)

        G_loss_grads = tape.gradient(G_loss, self.G.trainable_variables)
        self.G_opt.apply_gradients(zip(G_loss_grads, self.G.trainable_variables))   

        # Setup Gs's weights
        self.Gs.setup_as_moving_average_of(self.G)
        return G_loss, D_loss


    def create_summary(self):
        """ Create metrics and tensorboard summary writer. """
        import datetime
        self.G_loss_metrics = Mean()
        self.D_loss_metrics = Mean()
        self.summary_writer = tf.summary.create_file_writer(
            'log/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))


    def update_summary(self, metrics, step, time):
        """ Update losses with summary and print the training info """
        G_loss, D_loss = metrics

        self.G_loss_metrics(G_loss)
        self.D_loss_metrics(D_loss)

        with self.summary_writer.as_default():
            tf.summary.scalar('G_loss', self.G_loss_metrics.result(), step=step)
            tf.summary.scalar('D_loss', self.D_loss_metrics.result(), step=step)

        print(f'Step: {step}, ',
              f'Time: {time}, ',
              f'G_loss: {self.G_loss_metrics.result():.2f}, ',
              f'D_loss: {self.D_loss_metrics.result():.2f}, ')