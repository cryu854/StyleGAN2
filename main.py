""" USAGE
python main.py train --dataset_name afhq --dataset_path ./path/to/afhq_dataset_dir --batch_size 4 --res 512 --config e

python main.py inference --ckpt ./path/to/trained_checkpoint_dir --res 512 --num_labels 3 --config e
 """
import os
import argparse

from train import Trainer
from inference import Inferencer


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='StyleGAN2')
    parser.add_argument('command',help="'train' or 'inference'", type=str, choices=['train', 'inference'])
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'", type=str, default='ref', choices=['ref','cuda'])
    parser.add_argument('--config', help="Model's config be one of: 'e', 'f'", type=str, default='f')
    parser.add_argument('--dataset_name', help="Specific dataset be one of: 'ffhq', 'afhq', 'custom'", type=str, default='afhq', choices=['ffhq','afhq','custom'])
    parser.add_argument('--dataset_path', help='Dataset directory', default='./../../datasets/afhq/train_labels')
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=4)
    parser.add_argument('--res', help='Resolution of image', type=int, default=1024)
    parser.add_argument('--total_img', help='Training length of images', type=int, default=25000000)
    parser.add_argument('--ckpt', help='Checkpoints directory', type=str, default='./checkpoint')
    parser.add_argument('--num_labels', help='Number of labels', type=int, default=0)
    parser.add_argument('--label', help='Inference label', type=int, default=0)
    parser.add_argument('--truncation_psi', help='Inference truncation psi', type=float, default=0.5)
    parser.add_argument('--mode', help="Inference mode be one of: 'example', 'gif', 'mixing'", type=str, default='example', choices=['example','gif','mixing'])
    args = parser.parse_args()


    # Validate arguments
    if args.command == 'train':
        assert os.path.exists(args.dataset_path), 'Error: Dataset does not exist.'
        assert args.batch_size > 0
        assert args.res >= 4
        assert args.total_img > 0

        parameters = {
            'resolution' : args.res,
            'config' : args.config,
            'batch_size' : args.batch_size,
            'total_img' : args.total_img,
            'dataset_name' : args.dataset_name,
            'dataset_path' : args.dataset_path,
            'checkpoint_path' : args.ckpt,
            'impl' : args.impl
            }

        trainer = Trainer(**parameters)
        trainer.train()


    elif args.command == 'inference':
        assert os.path.exists(args.ckpt), 'Error: Checkpoint does not exist.'
        assert 0.0 <= args.truncation_psi <= 1.0, 'Error: Inference truncation_psi needs to be between 0 and 1.'
        assert args.res >= 4
        assert args.num_labels >= 0
        assert 0 <= args.label <= max(0, args.num_labels-1)

        parameters = {
            'resolution' : args.res,
            'num_labels' : args.num_labels,
            'config' : args.config,
            'truncation_psi' : args.truncation_psi,
            'checkpoint_path' : args.ckpt,
            }

        inferencer = Inferencer(**parameters)
        if args.mode == 'example':
            inferencer.genetate_example(num_example=10, label=args.label)
        elif args.mode == 'gif':
            inferencer.generate_gif(label=args.label) 
        elif args.mode == 'mixing':
            inferencer.style_mixing_example(row_seeds=[85,112,65,1188], col_seeds=[10,821,1789,293], col_styles=[0,1,2,3,4,5,6], label=args.label)


    else:
        print('Example usage : python main.py inference --ckpt ./afhq/checkpoint_512x512 --res 512 --num_labels 3 --config e')
        

if __name__ == '__main__':
    main()