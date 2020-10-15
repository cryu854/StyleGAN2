# StyleGAN2
"Analyzing and Improving the Image Quality of StyleGAN" in Tensorflow 2


### Todo
- [ ] Add inference_official_weights.py to exploit official trained model.
- [ ] Add metrics.py to compute PPL and FID.
- [ ] Train a model based on custom dataset with DiffAugment method.


### Requirements
You will need the following to run the above:
- TensorFlow = 2.3
- Python 3, Pillow 7.0.0, Numpy 1.18


### Attributions/Thanks
- Most of the code/CUDA are based on the [official implementation](https://github.com/NVlabs/stylegan2).
- The code of modules/DiffAugment_tf.py is from [data-efficient-gans](https://github.com/mit-han-lab/data-efficient-gans)
- The AFHQ training dataset is from [stargan-v2](https://github.com/clovaai/stargan-v2)