""" USAGE
python cal_metrics.py --ckpt ./weights-ffhq/official_1024x1024 --res 1024 --config f --num_labels 0 --mode ppl
python cal_metrics.py --ckpt ./weights-ffhq/official_1024x1024 --res 1024 --config f --num_labels 0 --mode fid --dataset './../datasets/ffhq_1024x1024'
 """
import os
import time
import argparse
import tensorflow as tf

from modules.metrics import FID, PPL 
from modules.generator import generator


def get_generator(checkpoint_path, resolution, num_labels, config, randomize_noise):
    Gs = generator(resolution, num_labels, config, randomize_noise=randomize_noise)
    ckpt = tf.train.Checkpoint(generator_clone=Gs)
    print(f'Loading network from {checkpoint_path}...')
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    return Gs


def calculate_metric(generator, num_labels, mode, dataset_path):
    fid50k_full_parameters = {'num_images':50000, 'num_labels':num_labels , 'batch_size':8}
    ppl_wend_parameters =  {'num_images':50000, 'num_labels':num_labels, 'epsilon':1e-4, 'space':'w', 'sampling':'end', 'crop':False, 'batch_size':2}

    if mode == 'fid':
        assert os.path.exists(dataset_path), 'Error: Dataset does not exist.'
        fid = FID(**fid50k_full_parameters)
        dist = fid.evaluate(generator, real_dir=dataset_path)
    else: # mode == 'ppl'
        ppl = PPL(**ppl_wend_parameters)
        dist = ppl.evaluate(generator)
    return dist


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate quality metric for StyleGAN2')
    parser.add_argument('--ckpt', help='Checkpoints directory', type=str, default='./checkpoint')
    parser.add_argument('--dataset', help='Dataset/pre-calculated statistics directory for FID evaluation', type=str)
    parser.add_argument('--res', help='Resolution of image', type=int, default=1024)
    parser.add_argument('--config', help="Model's config be one of: 'e', 'f'", type=str, default='f')
    parser.add_argument('--num_labels', help='Number of labels', type=int, default=0)
    parser.add_argument('--mode', help="Metric to evaluate the model be one of: 'fid', 'ppl'", type=str, default='fid')
    args = parser.parse_args()

    assert os.path.exists(args.ckpt), 'Error: Checkpoint does not exist.'
    assert args.mode.lower() in ['fid','ppl'], "Error: Metric mode needs to be one of: 'fid', 'ppl'."
    assert args.res >= 4
    assert args.num_labels >= 0


    Gs_parameters = {
            'checkpoint_path' : args.ckpt,
            'resolution' : args.res,
            'num_labels' : args.num_labels,
            'config' : args.config,
            'randomize_noise' : False,
            }
    Gs = get_generator(**Gs_parameters)

    start = time.perf_counter()
    dist = calculate_metric(Gs, args.num_labels, args.mode.lower(), args.dataset)
    print(f'{args.mode} : {dist:.3f}')
    print(f'Time taken for evaluation: {round(time.perf_counter()-start)}s')

if __name__ == '__main__':
    main()