## input image and output residul maps
## functionalized or modularized
## input image -> load model -> output image -> residul maps

import numpy as np
import torch
from models import Encoder, Decoder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import cv2
from PIL import Image

INPUT_SHAPE = [3, 128, 128]
    
class Predictor():
    def __init__(self, method: str, encoder_path: str, decoder_path: str,
                 input_shape=INPUT_SHAPE, zdim=32, dense=True, n_blocks=5) -> None:
        self.input_shape = input_shape
        self.zdim = zdim
        self.dense = dense
        self.n_blocks = n_blocks
        
        self.E = Encoder(method, fin=self.input_shape[0], zdim=self.zdim, dense=self.dense, n_blocks=self.n_blocks,
                         spatial_dim=self.input_shape[1]//2**self.n_blocks)
        self.Dec = Decoder(fin=self.zdim, nf0=self.E.backbone.nfeats//2, n_channels=self.input_shape[0],
                           dense=self.dense, n_blocks=self.n_blocks, spatial_dim=self.input_shape[1]//2**self.n_blocks)
        self.E.load_state_dict(torch.load(encoder_path))
        self.Dec.load_state_dict(torch.load(decoder_path))
        self.E.cuda()
        self.Dec.cuda()
        
    def predict(self, x: torch.Tensor):
        x_n = torch.tensor(x).cuda().float()
        # x_n = x_n.repeat(1, 3, 1, 1)
        z, z_mu, z_logvar, _ = self.E(x_n)
        xhat, _ = self.Dec(z)
        
        x_n = x_n.squeeze().permute([1, 2, 0]).cpu().detach().numpy()
        xhat = torch.sigmoid(xhat).squeeze().permute([1, 2, 0]).cpu().detach().numpy()
        x_g = np.mean(x_n, axis=-1)
        xhat = np.mean(xhat, axis=-1)
        residual_map = (abs(x_g - xhat))
        # resudial_map = resudial_map.squeeze().permute([1,2,0]).cpu().detach().numpy()
        return x_g, xhat, residual_map


encoder_path = 'vae_encoder_weights.pth'
decoder_path = 'vae_decoder_weights.pth'
device = torch.device("cuda")
model = Predictor('vae', encoder_path, decoder_path)


image_paths = 'nodule_anomaly_split/positive'
TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(INPUT_SHAPE[-1]),
            transforms.ToTensor(),
        ])

# Loading Data
image_paths = [f'nodule/{fname}' for fname in os.listdir('nodule')]

for i, image_path in enumerate(image_paths):
    image = cv2.resize(cv2.imread(image_path)[:, :, ::-1], (128, 128))
    torch_input = torch.tensor(image.transpose(2, 0, 1)/255.).unsqueeze(0)
    x_g, xhat, residual_map = model.predict(torch_input)
    plt.figure()
    plt.subplot(131)
    plt.imshow(x_g, cmap='gray')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(xhat, cmap='gray')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(residual_map > 0.2, cmap='gray')
    plt.axis('off')
    plt.show()
