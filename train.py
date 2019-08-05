from tqdm import tqdm
import numpy as np
import glob
import cv2
from utils import *
from model import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:",device)

BATCH_SIZE = 512
LATENT_DIM = 512

## LOSS
loss = nn.BCELoss()

## MODEL
generator = Generator(latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

generator.apply(weights_init)
discriminator.apply(weights_init)

print("..............Generator..............")
print(generator)
print("..............Discriminator..............")
print(discriminator)

## LOADING DATA
X_train = np.array(glob.glob("CAT*/*.jpg"))
y_train = np.ones((len(X_train), 1))

train_dataset = DataGenerator(X_train, y_train)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, num_workers=4)

## Ogptimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=2e-3, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=2e-3, betas=(0.5, 0.999))

def train(generator, discriminator, adversarial_loss, optimizer_G, optimizer_D, train_dataloader, epoch):
    print(f"Epoch: {epoch+1}/100")
    for epoch_ in range(epoch):
        running_loss_g = 0
        running_loss_d = 0
        tk0 = tqdm(train_dataloader, total=int(len(train_dataloader)))
        for i, sample_batched in enumerate(tk0):
            
            ## Sample Batch
            real_imgs = sample_batched[0].to(device, dtype=torch.float)
            real_label = sample_batched[1].to(device, dtype=torch.float)
            fake_label = torch.tensor(np.zeros((real_label.size()[0], 1))).to(device, dtype=torch.float)
            
            ## Generate Noise
            z = torch.tensor(np.random.normal(0, 1, (real_label.size()[0], LATENT_DIM))).to(device, dtype=torch.float)

            gen_imgs = generator(z)

            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), real_label)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss)/2
            running_loss_d += d_loss.item()
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            
            g_loss = adversarial_loss(discriminator(gen_imgs), real_label)
            running_loss_g += g_loss.item()
            g_loss.backward()
            optimizer_G.step()
            
            tk0.set_postfix(g_loss=(running_loss_g / (i+1)), d_loss=(running_loss_d / (i+1)))
            
            if (i+1)%10==0:
                fig = plt.figure(figsize=(10,7))
                for i , img in enumerate(gen_imgs[:5], 1):
                    img = img.cpu().detach().numpy().transpose((1, 2, 0))
                    img = 0.5 * img + 0.5
                    img = np.clip(img, 0, 1)
                    plt.axis('off')
                    plt.subplot(1, 5, i)
                    plt.imshow(img)
                plt.axis('off')
                plt.show()

if __name__ == '__main__':
    train(generator=generator, discriminator=discriminator, adversarial_loss=loss, optimizer_G=optimizer_G, optimizer_D=optimizer_D, train_dataloader=train_dataloader, epoch=100)