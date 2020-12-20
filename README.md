# StyleGAN2
This is an implementation of [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) and [Differentiable Augmentation for Data-Efficient GAN Training](https://arxiv.org/abs/2006.10738) in Tensorflow 2.3.

<div align = 'center'>
  <img src = 'results/gif/test_ffhq.gif' height = '240px'>
  <img src = 'results/gif/test_afhq.gif' height = '240px'>
</div>


## Style mixing examples

Check the [./results](https://github.com/cryu854/StyleGAN2/tree/main/results) folder to see more images.

<div align = 'center'>
  <img src='results/mixing/ffhq.png' height = '380px'>
  <img src='results/mixing/cat.png' height = '380px'>
</div>

## Training
Use `main.py` to train a StyleGAN2 based on given dataset.
Training takes 80s(CUDA op)/110s(Tensorflow op) for 100 steps(batch_size=4) on a GTX 1080ti.

Example usage for training on afhq-dataset:
```
python main.py train --dataset_name afhq                       \
                     --dataset_path ./path/to/afhq_dataset_dir \
                     --batch_size 4                            \
                     --res 512                                 \
                     --config e                                \
                     --impl ref                                \
```


## Inference
### Generate image_example/transition_gif/style_mixing_example
Use `main.py` to do inference on different mode and a given label.
Inference mode be one of: [example, gif, mixing].
**The pre-trained ffhq/afhq weights are [located here](https://drive.google.com/drive/folders/1LSEcdabnhDoJYLc3CkKjWVN6rBPnoOq4?usp=sharing).**

Example usage:
```
python main.py inference --ckpt ./weights-ffhq/official_1024x1024  \
                         --res 1024                                \
                         --config f                                \
                         --truncation_psi 0.5                      \
                         --mode example                            \
```


## Metric
### Calculate quality metric for StyleGAN2
Use `cal_metrics.py` to calculate PPL/FID score.
The pre-trained [LPIPS](https://arxiv.org/abs/1801.03924)'s weights(standard metric to estimate perceptual similarity) used in PPL will be downloaded automatically from [here](https://drive.google.com/drive/folders/1LSEcdabnhDoJYLc3CkKjWVN6rBPnoOq4?usp=sharing).

Evaluation time and results for the pre-trained FFHQ generator using one GTX 1080ti. 
| Metric    | Time      | Result   | Description
| :-----    | :---      | :-----   | :----------
| fid50k    | 1.5 hours | 3.096    | [Fr&eacute;chet Inception Distance](https://arxiv.org/abs/1706.08500) using 50,000 images.
| ppl_wend  | 2.5 hours | 144.044  | Perceptual Path Length for endpoints in *W*.

Example usage for FID evaluation:
```
python cal_metrics.py --ckpt ./weights-ffhq/official_1024x1024  \
                      --res 1024                                \
                      --config f                                \
                      --mode fid                                \
                      --dataset './datasets/ffhq'               \
```


## Todo
- [x] Add FFHQ official-weights inference feature.
- [x] Add metrics.py to compute PPL and FID.
- [ ] Train a model based on custom dataset with DiffAugment method.


## Requirements
You will need the following to run the above:
- TensorFlow = 2.3
- Python 3, Pillow 7.0.0, Numpy 1.18


## Attributions/Thanks
- Most of the code/CUDA are based on the [official implementation](https://github.com/NVlabs/stylegan2).
- The code of modules/DiffAugment_tf.py is from [data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans).
- The AFHQ training dataset is from [stargan-v2](https://github.com/clovaai/stargan-v2).
- The pre-trained FFHQ generator's weights are convered from [stylegan2-ffhq-config-f.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/stylegan2-ffhq-config-f.pkl).
- The pre-trained LPIPS's weights used in PPL are converted from [vgg16_zhang_perceptual.pkl](https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2).
