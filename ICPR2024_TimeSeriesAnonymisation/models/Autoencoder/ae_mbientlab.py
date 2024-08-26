import torch
import torch.nn.functional as F
import torch.nn as nn

class AutoencoderExtended(nn.Module):
    
    filter_dim = 4
    filter_dim_deconv = filter_dim
    stride_dim = 2
    padding_dim = (5, 0)
    
    def __init__(self):
        super(AutoencoderExtended, self).__init__()
        
        # Encoder (baseline image)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, self.filter_dim, self.stride_dim, self.padding_dim, padding_mode='replicate',bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, self.filter_dim, self.stride_dim, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            
            nn.Conv2d(32, 64, self.filter_dim, self.stride_dim, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )
        
        # Bottleneck (combination of latent and encoded features)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
            nn.ReLU(),        
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, self.filter_dim_deconv, self.stride_dim, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(32, 16, self.filter_dim_deconv, self.stride_dim, 0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),        
            nn.Dropout(p=0.1),
            
            nn.ConvTranspose2d(16, 1, self.filter_dim_deconv, self.stride_dim, (5, 0), bias=False),      
            nn.Sigmoid()  
        )
        
    def decode(self, x):
        dec = self.decoder(x)    
        return dec
    
    def encode(self, x):
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        return bottle

    def forward(self, x):
        # Encode the baseline image
        enc = self.encoder(x)
        # Bottleneck
        bottle = self.bottleneck(enc)
        # Decode
        dec = self.decoder(bottle)               
        return dec