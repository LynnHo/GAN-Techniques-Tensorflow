### Mnist GAN
```console
# dcgan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode none --norm batch_norm --model conv_mnist --dataset mnist --experiment_name conv_mnist_loss{gan}_gp{none}_norm{batch_norm}_wnorm{none}
# gan + dragan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode dragan --norm layer_norm --model conv_mnist --dataset mnist --experiment_name conv_mnist_loss{gan}_gp{dragan}_norm{layer_norm}_wnorm{none}
```

### Mnist WGAN
```console
# wgan + wgan-gp
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode wgan-gp --norm layer_norm --model conv_mnist --dataset mnist --experiment_name conv_mnist_loss{wgan}_gp{wgan-gp}_norm{layer_norm}_wnorm{none}
```

### CelebA GAN
```console
# dcgan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode none --norm batch_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{none}_norm{batch_norm}_wnorm{none}
# vgan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode none --norm batch_norm --vgan --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{none}_norm{batch_norm}_wnorm{none}_vgan
# gan + dragan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode dragan --norm layer_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{dragan}_norm{layer_norm}_wnorm{none}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode dragan --norm instance_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{dragan}_norm{instance_norm}_wnorm{none}
# gan + wgan-gp
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode wgan-gp --norm layer_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{wgan-gp}_norm{layer_norm}_wnorm{none}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode gan --gp_mode wgan-gp --norm instance_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{gan}_gp{wgan-gp}_norm{instance_norm}_wnorm{none}
```

### CelebA WGAN
```console
# wgan + weight_clip (does not work)
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --n_d_pre 100 --optimizer rmsprop --loss_mode wgan --gp_mode none --norm batch_norm --weights_norm weight_clip --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{none}_norm{batch_norm}_wnorm{weight_clip}
# wagn + wgan-gp
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode wgan-gp --norm layer_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{wgan-gp}_norm{layer_norm}_wnorm{none}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode wgan-gp --norm instance_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{wgan-gp}_norm{instance_norm}_wnorm{none}
# wgan + spectral_norm (does not work)
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode none --norm none --weights_norm spectral_norm --model conv_64 --lr_d 0.00005 --lr_g 0.00005 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{none}_norm{none}_wnorm{spectral_norm}
# wgan + wgan-gp + spectral_norm
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 5 --n_g 1 --loss_mode wgan --gp_mode wgan-gp --norm none --weights_norm spectral_norm --model conv_64 --lr_d 0.00005 --lr_g 0.00005 --dataset celeba --experiment_name conv64_celeba_loss{wgan}_gp{wgan-gp}_norm{none}_wnorm{spectral_norm}
```

### CelebA Hinge Loss
```console
# hinge
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode none --norm none --weights_norm none --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{none}_norm{none}_wnorm{none}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode none --norm batch_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{none}_norm{batch_norm}_wnorm{none}
# hinge + dragan
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode dragan --norm instance_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{dragan}_norm{instance_norm}_wnorm{none}
# hinge + wgan-gp
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode wgan-gp --norm instance_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{wgan-gp}_norm{instance_norm}_wnorm{none}
# hinge + none/dragan + spectral_norm
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode none --norm none --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{none}_norm{none}_wnorm{spectral_norm}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode dragan --norm none --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{dragan}_norm{none}_wnorm{spectral_norm}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode none --norm batch_norm --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{none}_norm{batch_norm}_wnorm{spectral_norm}
CUDA_VISIBLE_DEVICES=0 python train.py --n_d 1 --n_g 1 --loss_mode hinge --gp_mode dragan --norm instance_norm --weights_norm spectral_norm --model conv_64 --dataset celeba --experiment_name conv64_celeba_loss{hinge}_gp{dragan}_norm{instance_norm}_wnorm{spectral_norm}
```