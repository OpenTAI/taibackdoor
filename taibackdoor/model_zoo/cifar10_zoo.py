import torch
import requests
import os
import numpy as np
from taibackdoor.models.ResNet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from taibackdoor.attacks import *
from taibackdoor.datasets.utils import transform_options
from torchvision import transforms


cifar10_model_dict = {
    "ResNet18-BadNets-PR0.1":{
        "model": ResNet18,
        "model_weights": "13KplupQWymNVeGtDbG-tsprec6Lk_K09",
        "model_args": {
            "num_classes": 10,
        },
        "dataset": BadNetCIFAR10,
        "train_poison_idx": "1hH55eWQRF0ms0kVnquPLhINvOWOjO_fs",
        "test_poison_idx": "1ubY_ENJ81KsfxR6K3PPx9eFEyT1hT-Gv",
        "dataset_args": {
            "poison_rate": 0.1,
        },
        "train_transform": "NoAug",
        "test_transform": "NoAug",
    },
}

def download_gdrive(gdrive_id, fname_save):
    """ source: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url """

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, fname_save):
        CHUNK_SIZE = 32768

        with open(fname_save, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    print('Download started: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

    url_base = "https://docs.google.com/uc?export=download&confirm=t"
    session = requests.Session()

    response = session.get(url_base, params={'id': gdrive_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': gdrive_id, 'confirm': token}
        response = session.get(url_base, params=params, stream=True)

    save_response_content(response, fname_save)
    session.close()
    print('Download finished: path={} (gdrive_id={})'.format(
        fname_save, gdrive_id))

def download_google_drive_data(name, path, download=True):
    path = os.path.join(path, name)
    if not os.path.exists(path):
        os.makedirs(path)
    if download:
        download_gdrive(cifar10_model_dict[name]["model_weights"], os.path.join(path, "model_weights.pt"))
        download_gdrive(cifar10_model_dict[name]["train_poison_idx"], os.path.join(path, "train_poison_idx.npy"))
        download_gdrive(cifar10_model_dict[name]["test_poison_idx"], os.path.join(path, "test_poison_idx.npy"))
    return path

def load_cifar10_model_and_data(name, path='./data/', download=True):
    # Download model weights and data
    path = download_google_drive_data(name, path, download=download)

    # Load Model
    model_state_dict_path = os.path.join(path, "model_weights.pt")
    model = cifar10_model_dict[name]["model"](**cifar10_model_dict[name]["model_args"])
    model.load_state_dict(torch.load(model_state_dict_path, map_location='cpu'))
    
    # Load Data
    train_tf = transform_options[cifar10_model_dict[name]["train_transform"]]['train_transform']
    test_tf = transform_options[cifar10_model_dict[name]["test_transform"]]['test_transform']
    train_tf = transforms.Compose(train_tf)
    test_tf = transforms.Compose(test_tf)
    train_poison_idx = np.load(os.path.join(path, "train_poison_idx.npy"))
    test_poison_idx = np.load(os.path.join(path, "test_poison_idx.npy"))
    train_data = cifar10_model_dict[name]["dataset"](root=path, train=True, transform=train_tf, download=True,
                                             poison_idx=train_poison_idx, **cifar10_model_dict[name]["dataset_args"])
    test_data = cifar10_model_dict[name]["dataset"](root=path, train=False, transform=test_tf, download=True,
                                                    **cifar10_model_dict[name]["dataset_args"])
    
    resutls = {
        "train_data": train_data,
        "train_poison_idx": train_poison_idx,
        "test_data": test_data,
        "test_poison_idx": test_poison_idx,
        "model": model,
    }
    return resutls
