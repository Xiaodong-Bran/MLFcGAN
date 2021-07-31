

PyTorch implementation of [MLFcGAN: Multilevel Feature Fusion-Based Conditional GAN for Underwater Image Color Correction
](https://ieeexplore.ieee.org/document/8894129).

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by [Phillip Isola](https://github.com/phillipi) et al.


## Prerequisites

+ Linux
+ Python, Numpy, PIL
+ pytorch 1.2.0
+ torchvision 0.4.0

## Getting Started

+ Clone this repo:

    git clone git@github.com:Xiaodong-Bran/MLFcGAN.git
    
    cd MLFcGAN

+ download the pre-trained model: [google-dirve] (https://drive.google.com/open?id=1OREuAj6DplD0-ipQ3s37aZ6j9Q5kXvtO)

+ Modifiy the test_img_folder and test_output_path in test.sh The structure of the image folders should follows:

+-- name_of_the_dataset
|   +--source2target
    | +-- train
        | +-- source
        | +-- target
    | +-- test
        | +-- source
        | +--target
+ To test the model, please run:

    sh test.sh

+ To train the model, please run:

    sh train.sh

## Acknowledgments

This code is inspired by [pix2pix](https://phillipi.github.io/pix2pix/).

Highly recommend the more sophisticated and organized code [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) by [Jun-Yan Zhu](https://github.com/junyanz).
