#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import block

import os
import json
import time
import random
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as trans
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torchvision.utils as vutils
import torchvision.models as models


from PIL import Image


from evaluator import evaluation_model

seed = 5487
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)


# In[2]:


batch_size = 64
latent_size = 512
num_epochs = 500
num_imgchannels = 3
img_size = 64
fmap_size = 64
lr = 0.0005
beta = 0.5 # Adam Optimizer


# In[3]:


data_root = './dataset'
label_dict = json.loads(open('./objects.json', 'r').read())
print(label_dict)


def read_image(path, mode):
    with open(path, 'r') as data:
        raw_data = json.loads(data.read())
    
    clean_data = list()
    if mode == 'train':
        colors = 0
        shapes = 0
        for img_name, val in raw_data.items():
            object_list = list()
            for object_info in val:
                object_list.append(object_info)
            clean_data.append((img_name, object_list))
        
        return clean_data
    else:
        for img in raw_data:
            obj_list = list()
            for obj in img:
                obj_list.append(obj)
            
            clean_data.append(obj_list)
        return clean_data

    
def parse_label(raw_data):
    label = [0] * 24
    for obj in raw_data:
        label[label_dict[obj]] = 1

    return torch.FloatTensor(label)    


class CLEVR_Dataset(data.Dataset):
    def __init__(self, mode='Train', root='./dataset', device=None):
        
        if mode == 'Train':
            self.data = read_image('./train.json', 'train')
        else:
            self.data = read_image('./test.json', 'train')
            
        self.device = device
        self.root = root
        self.mode = mode
        self.transform = trans.Compose([
            trans.Resize(size=img_size),
            trans.CenterCrop(fmap_size),
#             trans.RandomRotation(degrees=15, fill=0),
            trans.ToTensor(),
            trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path = os.path.join(self.root, self.data[index][0])
        label = parse_label(self.data[index][1])
        
        img = Image.open(path).convert('RGB')
        img = self.transform(img).float()
        if self.device is not None:
            img = img.to(self.device)
            label = label.to(device)
        
        return img, label
        


# In[4]:


class Generator(nn.Module):
    def __init__(self, latent_size=latent_size):
        super(Generator, self).__init__()
        self.embed = nn.Linear(24, 128)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( latent_size + 128, fmap_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmap_size * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(fmap_size * 8, fmap_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d( fmap_size * 4, fmap_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d( fmap_size * 2,fmap_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d( fmap_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        return
    
    def forward(self, input, label):
        bs = input.shape[0]
        y = self.embed(label).view(bs, 128, 1, 1)
        y_ = nn.functional.relu(y)
        x = torch.cat((input, y_), 1)
        x = self.main(x)
        
        return x


# In[5]:


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embedding = nn.Linear(24, 64 * 64)
        self.projection = nn.Linear(24, 8192)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(4, fmap_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(fmap_size, fmap_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(fmap_size * 2, fmap_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(fmap_size * 4, fmap_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(512 * 4 * 4, 2048)),
            nn.utils.spectral_norm(nn.Linear(2048, 64)),
            nn.utils.spectral_norm(nn.Linear(64, 1)),
#             nn.Linear(512 * 4 * 4, 2048),
#             nn.Linear(2048, 64),
#             nn.Linear(64, 1),
        )
        
    def forward(self, input, img_label):
        bs = input.shape[0]
        
        y = self.embedding(img_label)
        y = nn.functional.relu(y).view(batch_size, 1, 64, 64)
        x = torch.cat((input, y), 1)
        x = self.main(x).view(bs, -1)
#         x = torch.sum(x, dim=(2, 3))
        
        d = self.projection(img_label)
        x = self.classifer(x) + torch.sum(x * d, dim=(1), keepdim=True)
            
        return x


# In[6]:


# From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#weight-initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def feature_extract(extractor, img):
    
    return extractor.features(img)


# In[7]:


def train_iters(net_G, net_D, eval_model, num_epochs=num_epochs, lr=lr, print_every=50, start_epoch=0):
    # https://pytorch.org/docs/master/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
    criterion = nn.BCEWithLogitsLoss()
    
    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/srgan/srgan.py
    feature_criterion = nn.L1Loss()
    
    optim_G = optim.Adam(net_G.parameters(), lr=lr, betas=(beta, 0.999))
    optim_D = optim.Adam(net_D.parameters(), lr=lr, betas=(beta, 0.999))
    
    scheduler_G = optim.lr_scheduler.StepLR(optim_G, step_size=50, gamma=0.2)
    scheduler_D = optim.lr_scheduler.StepLR(optim_D, step_size=50, gamma=0.2)
    
    loss_G = list()
    loss_D = list()
    
    fake_label_True = torch.ones(batch_size, 1).to(device)
    fake_label_False = torch.zeros(batch_size, 1).to(device)
    
    epoch_len = len(training_set)
    net_G.train()
    net_D.train()
    
    feature_extractor = models.vgg19(pretrained=True).to(device)
    feature_extractor.eval()
    
    for epoch in range(1, num_epochs + 1):
        print('=' * 40)
        print(f'Epoch {epoch} Starts...\n')
        start = time.time()
        
        print_loss_D = 0
        print_loss_G = 0
        
        for i, data in enumerate(training_set):
            '''
            data[0]: bs X 3 X 64 X 64, img
            data[1]: bs X 24, label
            '''

            
            output = net_D(data[0], data[1].detach())
            D_X = output.mean().item()
            optim_D.zero_grad()
            loss_DReal = criterion(output, fake_label_True)
#             print(torch.mean(output * fake_label_True))
            loss_DReal.backward()

            noise = torch.randn(batch_size, latent_size, 1, 1, device=device)

            fake_in = net_G(noise, data[1].detach())
            fake_out = net_D(fake_in, data[1].detach())

            loss_DFake = criterion(fake_out, fake_label_False)
            loss_DFake.backward()
            loss_d = loss_DReal + loss_DFake
            optim_D.step()
            print_loss_D += loss_d.item()
            
            
            optim_G.zero_grad()
            noise = torch.randn(batch_size, latent_size, 1, 1, device=device)

            fake_in = net_G(noise, data[1].detach())
            out = net_D(fake_in, data[1].detach())
            D_GZ = out.mean().item()
            loss_g = criterion(out, fake_label_True) + 0.001 * feature_criterion(feature_extract(feature_extractor, fake_in), feature_extract(feature_extractor, data[0]))

            loss_g.backward()
            optim_G.step()

            print_loss_G += loss_g.item()
            
            if i % print_every == 0:
                print(f'\tLoss of G in this mini batch: {loss_g.item()}')
                print(f'\tLoss of D in this mini batch: {loss_d.item()}')
                
                print(f'\tEpoch {epoch}...{i * 100 / epoch_len}%\n')
        
        loss_G.append(print_loss_G / epoch_len)
        loss_D.append(print_loss_D / epoch_len)
        print(f'Average Loss of G in Epoch {epoch}: {loss_G[-1]}')
        print(f'Average Loss of D in Epoch {epoch}: {loss_D[-1]}\n')
        if epoch % 2 == 1:
            net_G.eval()
            fake_in = test(net_G, eval_model)
            plot_fake_imgs(fake_in)
            net_G.train()
            
        end = time.time()
        print(f'{end - start} seconds used.\n')
        
        
        if epoch % 50 == 0:
            print('Save Checkpoint at Epoch', epoch)
            torch.save(net_G.state_dict(), f'G_{epoch + start_epoch}.pth')
            torch.save(net_D.state_dict(), f'D_{epoch + start_epoch}.pth')
    

def test(net_G, eval_model):
    with torch.no_grad():
        noise = torch.randn(testing_set.shape[0], latent_size, 1, 1, device=device)
        fake_img = net_G(noise, testing_set)
        acc = eval_model.eval(fake_img, testing_set)
        print(f'Accuracy: {acc * 100} %')
        
        return fake_img
    
    
def plot_fake_imgs(img_tensors):
    fig = plt.figure(figsize=(16,16), dpi=300)
    plt.axis("off")
    ims = np.transpose(vutils.make_grid(
        (img_tensors.cpu().detach() + 1) / 2, padding=2), (1, 2, 0))
    plt.imshow(ims)
    
    plt.show()
    plt.close()
    
    return
    


# In[8]:


net_G = Generator().to(device)
net_D = Discriminator().to(device)

net_G.apply(weights_init)
net_D.apply(weights_init)

eval_model = evaluation_model()

training_set = CLEVR_Dataset(device=device)
training_set = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, drop_last=True)

testing_set = read_image('./test.json', mode='test')
testing_set = torch.stack([parse_label(test_img) for test_img in testing_set]).to(device)

print(net_G)
print(net_D)


# In[ ]:


train_iters(net_G, net_D, eval_model, start_epoch=0)


# In[ ]:





# In[ ]:


get_ipython().run_line_magic('debug', '')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




