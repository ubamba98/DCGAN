import torch.nn as nn
import torch

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, img_shape=(64, 64, 3), latent_dim=256):
        super(Generator, self).__init__()

        h, w, c = img_shape
        self.img_shape = img_shape

        self.ll = nn.Linear(latent_dim, 256*49)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=c, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh(),
        )

    def forward(self, z):
        out = self.ll(z)
        out = out.view(out.shape[0], 256, 7, 7)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(64, 64, 3)):
        super(Discriminator, self).__init__()
        
        h, w, c = img_shape
        def discriminator_block(in_filters, out_filters, bn=True):
          block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
          if bn:
              block.append(nn.BatchNorm2d(out_filters))
          block.append(nn.LeakyReLU(0.2, inplace=True))
          return block

        self.features = nn.Sequential(
            *discriminator_block(c, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(h*w//2, 1), 
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.features(img)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out
      
    