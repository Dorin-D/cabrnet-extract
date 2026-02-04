import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cabrnet.archs.generic.model import CaBRNet
import os
import numpy as np

from cabrnet.core.utils.data import DatasetManager


dataset_path = "./configs/protopnet/cub200/dataset.yml"
# dataset_path = "./configs/protopnet/cub200/dataset_smaller_batch.yml"
dataloaders = DatasetManager.get_dataloaders(config=dataset_path, sampling_ratio=1)

train_loader = dataloaders['train_set']
test_loader = dataloaders['test_set']

model_arch = "src/counterfactuals/intermediate_model_arch.yml"
model_state_dict = "trained_models/protopnet_cub200_resnet50_s1337/final/model_state.pth"
model_classifier = CaBRNet.build_from_config(config=model_arch, state_dict_path=model_state_dict)



class LatentDiffusionModel(torch.nn.Module):
    def __init__(self, classifier):
        super(LatentDiffusionModel, self).__init__()
        self.classifier = classifier

    def extract_classifier_features(self, x):
        return self.classifier.extractor(x)['convnet'] # should return latent features of shape (b_size, 256, 7, 7)

    def forward(self, x):
        # x : input image
        latent_features = self.extract_classifier_features(x)
        data = (x, latent_features)
        # do stuff
        pass    
        return data

device = 'cuda' if torch.cuda.is_available else 'cpu'

model = LatentDiffusionModel(model_classifier)
model.to(device)

for epoch in range(1):
    for i, data in enumerate(train_loader):
        img, cls = data
        img = img.to(device)
        forward_result = model(img)
        print(forward_result[0].shape, forward_result[1].shape)
        exit("Success")