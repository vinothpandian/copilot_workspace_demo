import os
import torch
import torchvision

def download_mnist():
    data_dir = 'data/MNIST'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    torchvision.datasets.MNIST(root=data_dir, download=True)
