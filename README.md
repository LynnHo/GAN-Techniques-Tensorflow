# <p align="center"> GANs: Losses, Regularizations and Normalizations </p>

Tensorflow implementation of some common techniques of GANs, including losses, regularizations and normalizations.

## Techniques

- Losses:
    - [x] GAN
    - [x] LSGAN
    - [x] WGAN
    - [x] Hinge
- Gradient Regularizations:
    - [x] WGAN-GP
    - [x] DRAGAN
- Weights Normalizations/Regularizations:
    - [x] SpectralNorm
    - [x] Weight Clipping
    - [ ] WeightNorm
    - [ ] Orthonormal Regularization
- Normalizations:
    - [x] BatchNorm, InstanceNorm, LayerNorm

## Usage

- Prerequisites
    - Tensorflow 1.8
    - Python 2.7 or 3.6

- Training
    - Important Arguments (See the others in [train.py](train.py))
        - `--n_d`: # of d steps in each iteration (default: `1`)
        - `--n_g`: # of g steps in each iteration (default: `1`)
        - `--loss_mode`: gan loss (choices: `[gan, lsgan, wgan, hinge]`, default: `gan`)
        - `--gp_mode`: type of gradient penalty (choices: `[none, dragan, wgan-gp]`, default: `none`)
        - `--norm`: normalization (choices: `[batch_norm, instance_norm, layer_norm, none]`, default: `batch_norm`)
        - `--weights_norm`: weights normalization (choices: `[none, spectral_norm, weight_clip]`, default: `none`)
        - `--model`: model (choices: `[conv_mnist, conv_64]`, default: `conv_mnist`)
        - `--dataset`: dataset (choices: `[mnist, celeba]`, default: `mnist`)
        - `--experiment_name`: name for current experiment (default: `default`)
    - Examples (See more in [examples.md](examples.md))
        ```console
        # gan + dragan
        CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode dragan --norm layer_norm --model conv_mnist --dataset mnist --experiment_name conv_mnist_loss{gan}_gp{dragan}_norm{layer_norm}_wnorm{none}
        # hinge + spectral_norm
        CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode none --norm none --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{none}_norm{none}_wnorm{spectral_norm}
        # hinge + dragan + instance_norm + spectral_norm
        CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode dragan --norm instance_norm --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{dragan}_norm{instance_norm}_wnorm{spectral_norm}
        # wgan + wgan-gp + spectral_norm
        CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode wgan-gp --norm none --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{wgan-gp}_norm{none}_wnorm{spectral_norm}
        ```

## Datasets

1. CelebA should be prepared by yourself in ***./data/celeba/img_align_celeba/*.jpg***
    - Download the dataset: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0
    - the above links might be inaccessible, the alternatives are
        - ***img_align_celeba.zip***
            - https://pan.baidu.com/s/1eSNpdRG#list/path=%2FCelebA%2FImg or
            - https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg
2. Mnist will be automatically downloaded

## Exemplar Results

1. Hinge + DRAGAN + InstanceNorm + SpectralNorm
<p align="center"> <img src="pics\hinge+dragan+instance_norm+spectral_norm.jpg"> </p>