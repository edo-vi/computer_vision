# Deep Learning and Computer Vision Project

## Description

This is the codebase for my project in Computer Vision and Deep Learning. I've used Lightning (pyTorch) to implement the models (and the training, validation etc. facilities), gin as configuration framework (to inject some parameters in the classes, see config.cfg), and of course torchvision etc.

The main files are the following:

- train_denoising.ipynb: notebook to train the denoising models (contained in the files DenoisingAE.py and DenoisingResnet.py)
- DataLoader.py: file containing the utilies to load the dataset and applying noise to it (as torch.transform)
- make_datasets.ipynb: notebook to automate the creation of the denoised datasets
- results.ipynb: notebook to analyze and print the experimantal results.
- config.cfg: gin configuration file, used to inject parameters in classes annotated with ```@gin.configurable```
  
I thoroughly commented the files so as to be as clear as possible.

To make things easy, the notebook have already the results that I have used in the report. I'm also including the datasets.
