## input image and output residul maps
## functionalized or modularized
## input image -> load model -> output image -> residul maps

import numpy as np
import torch
from models import Encoder, Decoder
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image

INPUT_SHAPE = [3, 128, 128]
    
class Predictor():
    def __init__(self, method: str, encoder_path: str, decoder_path: str, input_shape = INPUT_SHAPE, zdim = 32, dense = True, n_blocks = 5) -> None:
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
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        x_n = torch.tensor(x).cuda().float()
        # x_n = x_n.repeat(1, 3, 1, 1)
        z, z_mu, z_logvar, _ = self.E(x_n)
        xhat, _ = self.Dec(z)
        
        x_n = x_n.squeeze().permute([1,2,0]).cpu().detach().numpy()
        xhat = xhat.squeeze().permute([1,2,0]).cpu().detach().numpy()
        xhat = (xhat - np.min(xhat)) / (np.max(xhat) - np.min(xhat))
        x_g = np.mean(x_n, axis=-1)
        xhat = np.mean(xhat, axis=-1)
        residual_map = (abs(x_g - xhat)*255).astype(np.uint8)
        # residual_map = abs(grayscale(xhat)-grayscale(x_n))
        # resudial_map = (np.clip(abs((x_n) - np.clip(xhat, 0, 1)), 0, 1)*255).astype(np.uint8)
        # resudial_map = resudial_map.squeeze().permute([1,2,0]).cpu().detach().numpy()
        
        return x_n.astype(np.uint8), xhat.astype(np.uint8), residual_map


encoder_path = 'test_model/vae_encoder_weights.pth'
decoder_path = 'test_model/vae_decoder_weights.pth'
device = torch.device("cuda")
model = Predictor('vae', encoder_path, decoder_path)


image_paths = 'nodule_anomaly_split/positive'
TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(INPUT_SHAPE[-1]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.38169734018417545, 0.35629716336767236, 0.3216164951658236],
            #                     std=[0.23080161182849712, 0.22390099134794886, 0.20951167832871959]),
            # transforms.Grayscale()
        ])

# Loading Data
test_data = datasets.ImageFolder(root=image_paths, transform=TRANSFORM_IMG)
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)

for i, (image, _) in enumerate(test_data_loader):
    x_n, xhat, residual_map = model.predict(image)
    # x_n = Image.fromarray(x_n)
    # xhat = Image.fromarray(xhat)
    residual_map = Image.fromarray(residual_map)
    residual_map.save(os.path.join('nodule_anomaly_split', 'new_imgs', f'{i}.jpg'), 'JPEG')

    




