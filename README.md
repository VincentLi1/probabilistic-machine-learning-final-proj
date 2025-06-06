# Probabilistic Methods for Overcoming Covariate Shift

## Reproduction

1. Clone this repository

```bash
git clone https://github.com/VincentLi1/probabilistic-machine-learning-final-proj.git
```

2. Create the conda environment from **environment.yml** and activate:

```bash
conda env create -f environment.yml
conda activate pml_env
```

Since torch does not nicely integrate with anaconda, you will likely need to install torch via pip separately:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Download the [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and [Tiny ImageNet-C](https://zenodo.org/records/2536630) datasets (described in detail below). Create a directory 'data' in the main directory and store the extracted datasets in this folder. These datasets are not included on GitHub due to size.

## Datasets

The training dataset used for both models is Tiny ImageNet-200. Tiny ImageNet is a dataset of 64x64 images classified into 200 different categories. The training, validation, and testing sets are all balanced between the categories. The train set has 100k images, the validation set has 10k images, and the test set has 10k images. The dataset can be downloaded [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip).

The dataset we will test the models on is the ImageNet-C dataset introduced in [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261) (ICRL 2019). This dataset consists of a number of images from the test set of ImageNet-200 with pertubations such as Gaussian noise, snow, jpeg compression, and motion blur applied. The images are 64x64 and are of 200 different categories, and there are 50 samples per category per transformation for a total of 10k test images for each pertubation. The dataset can be downloaded [here](https://zenodo.org/records/2536630).

## Credit

All code defining the SWAG class for tracking second moments of model parameters comes from the GitHub for "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (Maddox et al, 2019).



