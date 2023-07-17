# 17th july
import torch
import torchvision
from torchvision import datasets, transforms
import os
from torch import optim
import torch.nn as nn
from torch.nn import init
from IPython.display import Image
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil
import cv2
import numpy as np

json_dir = "/home/cv-gpu-1/sanya_workspace/IDD_dataset/json"
image_dir = "/home/cv-gpu-1/sanya_workspace/IDD_dataset/no_box"
generator_data = "/home/cv-gpu-1/sanya_workspace/IDD_dataset/generator/g_data"
discriminator_data = "/home/cv-gpu-1/sanya_workspace/IDD_dataset/discriminator"

if not os.path.isdir(generator_data):
    os.makedirs(generator_data)

if not os.path.isdir(discriminator_data):
    os.makedirs(discriminator_data)

for json_file in os.listdir(json_dir):
    json_file = os.path.join(json_dir,json_file)
    with open(json_file) as data_file:
        data = json.load(data_file)
        flags = data["flags"]

        imageName = data["imagePath"]
        imagePath = os.path.join(image_dir,imageName)
        if not os.path.isfile(imagePath):
            continue
        if flags["improper_prediction"] == True or flags["missing_prediction"] == True:
            shutil.copy(imagePath,(os.path.join(generator_data,imageName)))
        elif flags["correct_prediction"] == True:
            shutil.copy(imagePath,(os.path.join(discriminator_data,imageName)))
        else:
            continue
print("completed check data")

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5,), std=(0.5,))
                                 ])

# data for generator
g_data = "/home/cv-gpu-1/sanya_workspace/IDD_dataset/generator_2"
dataset_g = datasets.ImageFolder(g_data, transform=transform)
batch_size = 10
dataloader_g = torch.utils.data.DataLoader(dataset_g, batch_size, shuffle=True)

# data for the discriminator
d_data = "IDD_dataset/discriminator_2"
dataset_d = datasets.ImageFolder(d_data, transform=transform)
dataloader_d = torch.utils.data.DataLoader(dataset_d, batch_size = 10, shuffle=True)

# move to gpu if there
device = torch.device('cuda')

# to de normalize the images when viewing them
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# network for discriminator
image_size = 784
hidden_size = 224
input_size = 1505280

# pre trained model for discriminator
D = torchvision.models.wide_resnet50_2(pretrained=True)
for param in D.parameters():
    param.required_grad = False

num_ftrt = D.fc.in_features

D.fc = nn.Linear(num_ftrt,1)
D = D.cuda()

# network for generator

""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
        

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        

        """ Classifier """
        self.outputs = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        

        """ Classifier """
        outputs = self.outputs(d4)


        return outputs

inputs = torch.randn((2, 3, 256, 256)).to(torch.device('cuda'))
G = build_unet().cuda()

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

criterion = nn.CrossEntropyLoss() # loss function
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002) # optimizer for discriminator
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002) # optimizer for generator

# make the gradients for generator and discriminator to 0
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

# train discriminator
def train_discriminator(img_d, img_g):


    real_labels = torch.ones(img_d.shape[0],1).to(device) # labels for real image is 1
    fake_labels = torch.zeros(img_g.shape[0],1).to(device) # labels for fake image is 0


    # loss for real image
    real_outputs = D(img_d) # prediction from the discriminator
    loss_real = criterion(real_outputs, real_labels)
    

    # loss for fake images
    fake_images = G(img_g)
    fake_outputs = D(fake_images)
    loss_fake = criterion(fake_outputs, fake_labels)

    # update weights
    d_loss = loss_real + loss_fake # total loss
    reset_grad() # reset gradients
    d_loss.backward()  # compute gradients
    d_optimizer.step()  # adjust the parameters using backpropogation
    return d_loss

# train generator by using the discriminator
def train_generator(img_g):

    fake_images = G(img_g) # image generated by generator
    labels = torch.ones(img_g.shape[0],1).to(device) # to trick the discrimintor we assume the labels to be 1
    pred = D(fake_images) # labels calculated by the discriminator
    g_loss = criterion(pred, labels)
    
    reset_grad()
    g_loss.backward()
    g_optimizer.step()

    return g_loss, fake_images

# create a directory to save the images by the generator
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# to save the images by the generator
def save_fake_images(img_g, index):
    i = G(img_g)
    i = i.reshape(img_g.shape[0],3,224,224).cpu().detach().numpy()
    img = i[0]
    img = np.copy(img.transpose(-1,1,0))
    cv2.imwrite(str(index)+"_img.jpg",img)
 
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

init_weights(G)
G = G.cuda()


# train the generator and discriminator with 30 epochs
total_step = len(dataloader_g)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in tqdm(range(50)): 
    i = 0
    for img_g, a, in tqdm(dataloader_g):
        for img_d, b in (dataloader_d):
            i = i+1
            # load the images and make it to a vector
            img_d = img_d.cuda()
            img_g = img_g.cuda()
         
            # training
            d_loss = train_discriminator(img_d, img_g)
            g_loss, fake_images = train_generator(img_g)


            if (i+1) % 500 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

   # save the image
            save_fake_images(img_g, epoch+1)    

plt.plot(d_losses, '-')
plt.plot(g_losses, '-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Discriminator', 'Generator'])
plt.title('Losses');
