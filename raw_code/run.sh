#!/bin/bash
## SBATCH things

#python main_train.py --checkpoints_dir "./MNIST_full_ft" --target_label 0 --arch "Resnet18"  --batch_size 128 --resize 32  --dataset "MNIST" --quant_weight 0.5 #--optim 'sgd'
#python model_compare_ours.py --target_label 0 --batch_size 64 --dataset "TinyImagenet"
#python scaleUp_eva_ours_tiny.py
#python MM_BD.py --checkpoints_dir "$path"