import tensorflow as tf

from modules.DiffAugment_tf import DiffAugment


#----------------------------------------------------------------------------
# Final loss function from "Analyzing and Improving the Image Quality of StyleGAN" 
class ns_pathreg_r1:
    def __init__(self, G, D, batch_size, num_labels, pl_mean, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2.0):
        self.G = G
        self.D = D
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.pl_mean = pl_mean
        self.pl_denorm = tf.math.rsqrt(tf.cast(self.G.resolution*self.G.resolution, tf.float32))
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.gamma = 0.0002 * (self.G.resolution ** 2) / self.batch_size # heuristic formula


    # R1 and R2 regularizers from the paper
    # "Which Training Methods for GANs do actually Converge?", Mescheder et al. 2018
    @tf.function
    def get_D_loss(self, real_images, real_labels, compute_reg=False):
        latents = tf.random.normal([self.batch_size, 512])
        fake_images = self.G([latents, real_labels], training=True)
        fake_scores = self.D([fake_images, real_labels], training=True)
        
        reg = 0.0
        if compute_reg:
            with tf.GradientTape(watch_accessed_variables=False) as r1_tape:
                r1_tape.watch([real_images, real_labels])
                real_scores = self.D([real_images, real_labels], training=True)
                real_loss = tf.reduce_sum(real_scores)
            real_grads = r1_tape.gradient(real_loss, real_images)
            gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
            reg = gradient_penalty * (self.gamma * 0.5)
        else:
            real_scores = self.D([real_images, real_labels], training=True)
        
        loss = tf.math.softplus(fake_scores) + tf.math.softplus(-real_scores)
        loss = tf.reduce_mean(loss)

        return loss, reg


    # Non-saturating logistic loss with path length regularizer from the paper
    # "Analyzing and Improving the Image Quality of StyleGAN", Karras et al. 2019
    @tf.function
    def get_G_loss(self, real_images, real_labels, compute_reg=False):
        latents = tf.random.normal([self.batch_size, 512])
        fake_images = self.G([latents, real_labels], training=True)
        fake_scores = self.D([fake_images, real_labels], training=True)

        reg = 0.0
        if compute_reg:
            # Path length regularization.
            # Evaluate the regularization term using a smaller minibatch to conserve memory.
            pl_batch = tf.maximum(1, self.batch_size // self.pl_batch_shrink)
            pl_latents = tf.random.normal([pl_batch, 512])
            labels_indice = tf.random.uniform([pl_batch], 0, self.num_labels, dtype=tf.int32)
            pl_labels = tf.one_hot(labels_indice, self.num_labels) if self.num_labels > 0 else tf.zeros([pl_batch, 0])
            with tf.GradientTape(watch_accessed_variables=False) as pl_tape:
                pl_tape.watch([pl_latents, pl_labels])
                fake_images, pl_w = self.G([pl_latents, pl_labels], return_latents=True, training=True)
                # Compute |J*y|.
                pl_noise = tf.random.normal(tf.shape(fake_images)) * self.pl_denorm
                pl_noise_applied = tf.reduce_sum(fake_images * pl_noise)
            pl_grads = pl_tape.gradient(pl_noise_applied, pl_w)
            pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))

            # Track exponential moving average of |J*y|.
            new_pl_mean = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
            self.pl_mean.assign(new_pl_mean)

            # Calculate (|J*y|-a)^2.
            pl_penalty = tf.square(pl_lengths - self.pl_mean)
            
            # Apply weight.
            #
            # Note: The division in pl_noise decreases the weight by num_pixels, and the reduce_mean
            # in pl_lengths decreases it by num_affine_layers. The effective weight then becomes:
            #
            # gamma_pl = pl_weight / num_pixels / num_affine_layers
            # = 2 / (r^2) / (log2(r) * 2 - 2)
            # = 1 / (r^2 * (log2(r) - 1))
            # = ln(2) / (r^2 * (ln(r) - ln(2))
            #
            reg = pl_penalty * self.pl_weight
 
        loss = tf.math.softplus(-fake_scores)
        loss = tf.reduce_mean(loss)

        return loss, reg


#----------------------------------------------------------------------------
# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
class ns_DiffAugment_r1:
    def __init__(self, G, D, batch_size, policy='color,translation,cutout'):
        self.G = G
        self.D = D
        self.batch_size = batch_size
        self.policy = policy
        self.gamma = 0.0002 * (self.G.resolution ** 2) / self.batch_size # heuristic formula


    @tf.function
    def get_D_loss(self, real_images, real_labels, compute_reg=False):
        latents = tf.random.normal([self.batch_size, 512])
        fake_images = self.G([latents, real_labels], training=True)
        fake_scores = self.D([DiffAugment(fake_images, policy=self.policy), real_labels], training=True)

        reg = 0.0
        if compute_reg:
            with tf.GradientTape(watch_accessed_variables=False) as r1_tape:
                r1_tape.watch([real_images, real_labels])
                real_scores = self.D([DiffAugment(real_images, policy=self.policy), real_labels], training=True)
                real_loss = tf.reduce_sum(real_scores)
            real_grads = r1_tape.gradient(real_loss, real_images)
            gradient_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
            reg = gradient_penalty * (self.gamma * 0.5)
        else:
            real_scores = self.D([DiffAugment(real_images, policy=self.policy), real_labels], training=True)

        loss = tf.math.softplus(fake_scores) + tf.math.softplus(-real_scores)
        loss = tf.reduce_mean(loss)

        return loss, reg


    @tf.function
    def get_G_loss(self, real_images, real_labels, compute_reg=False):
        latents = tf.random.normal([self.batch_size, 512])
        fake_images = self.G([latents, real_labels], training=True)
        fake_scores = self.D([DiffAugment(fake_images, policy=self.policy), real_labels], training=True)

        loss = tf.math.softplus(-fake_scores)
        loss = tf.reduce_mean(loss)
        reg = 0.0
    
        return loss, reg