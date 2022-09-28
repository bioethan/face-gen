# Getting packages
# Importing modules and data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_vis
from torchvision import transforms, datasets, io
from torch import optim as optim
from torch.utils.data import TensorDataset, DataLoader
# for visualization
from matplotlib import pyplot as plt
import math
import numpy as np
import os
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import tqdm

# Global vars 
# Loading the data
TRIAL_NUM = 2
batch_size = 100
image_size = 256

gen_init_image = 32
num_gen_features = 250

gen_learning_rate =  0.0002
dis_learning_rate = 0.0002

# Hyperparameters
LATENT_SIZE = 200
NUM_EPOCHS = 2500
BETA = 0.5

# Paths for saving data
SAVE_PATH = '/home/ethanbrown/code/results/trial%d/' % TRIAL_NUM

# Modifying images as needed for the size of the neural network 
DATA_PATH = '/home/ethanbrown/data_general/data/'
list_data_dirs = os.listdir(DATA_PATH)
list_data_dirs = [x for x in list_data_dirs if x[0] != '.']
print('Number of Data Dirs is:', len(list_data_dirs))
print(list_data_dirs)
# Option to use large or small images:
option1 = 's.'
option2 = 'S.'

# Then transforming images
pic_iter = 0
for i in list_data_dirs:
   if 'BALDUR' in i:
      list_images = os.listdir(DATA_PATH + i + '/')
      list_images = [x for x in list_images if option1 in x or option2 in x] 
      for j in list_images:
         image = Image.open(DATA_PATH + i + '/' + j, mode='r')
         image.resize((image_size,image_size), Image.LANCZOS ).convert('RGB').save('./datahigh_jpg/' + 'picnum%d' % pic_iter  + '.jpg')
         pic_iter = pic_iter + 1
   elif 'uploads' in i:
      list_images = os.listdir(DATA_PATH + i + '/')
      list_images = [x for x in list_images if x[0] != '.']
      for j in list_images:
         image = Image.open(DATA_PATH + i + '/' + j, mode='r')
         image.resize((image_size,image_size), Image.LANCZOS ).convert('RGB').save('./datahigh_jpg/' + 'picnum%d' % pic_iter  + '.jpg')
         pic_iter = pic_iter + 1
   elif 'NEXUS' in i:
      list_dirs = os.listdir(DATA_PATH + i + '/')
      list_dirs = [x for x in list_dirs if x[0] != '.']
      for k in list_dirs:
         next_list_dirs = os.listdir(DATA_PATH + i + '/' + k + '/')
         next_list_dirs = [x for x in next_list_dirs if x[0] != '.']
         for l in next_list_dirs:
            final_list_dirs = os.listdir(DATA_PATH + i + '/' + k + '/' + l + '/')
            final_list_dirs = [x for x in final_list_dirs if x[0] != '.']
            for m in final_list_dirs:
               image = Image.open(DATA_PATH + i + '/' + k + '/' + l + '/' + m + '/Small.png' , mode='r')
               image.resize((image_size,image_size), Image.LANCZOS ).convert('RGB').save('./datahigh_jpg/' + 'picnum%d' % pic_iter  + '.jpg')
               pic_iter = pic_iter + 1
               image.close()
   elif 'more' in i:
      list_images = os.listdir(DATA_PATH + i + '/')
      list_images= [x for x in list_images if x[0] != '.']
      for j in list_images:
         image = Image.open(DATA_PATH + i + '/' + j, mode='r')
         image.resize((image_size,image_size), Image.LANCZOS ).convert('RGB').save('./datahigh_jpg/' + 'picnum%d' % pic_iter  + '.jpg')
         pic_iter = pic_iter + 1



# Using colab GPU for quick training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# DCGAN discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Convolutional block for image discrimination
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1), 
            nn.LeakyReLU(0.2), 

            nn.Conv2d(8, 16, 3, 2, 1), 
            nn.LeakyReLU(0.2), 
            nn.BatchNorm2d(16, 0.8),

            nn.Conv2d(16, 32, 3, 2, 1), 
            nn.LeakyReLU(0.2), 
            nn.BatchNorm2d(32, 0.8),

            nn.Conv2d(32, 64, 3, 2, 1), 
            nn.LeakyReLU(0.2), 
            nn.BatchNorm2d(64, 0.8),

            nn.Conv2d(64, 128, 3, 2, 1), 
            nn.LeakyReLU(0.2))
        # Classification layer
        self.class_layer = nn.Sequential(nn.Linear(128 * 4 * 4, 1), nn.Sigmoid())

    def forward(self, x):
        # Resulting convolution over the image
        out = self.conv_block(x)
        out = out.view(out.shape[0], -1)
        # print(out.size())
        class_out = self.class_layer(out)

        return class_out

# DCGAN generator class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Linear layer to take from the initial size to the 
        self.latent_reshape = nn.Linear(LATENT_SIZE, gen_init_image * gen_init_image * num_gen_features)

        # Convolutional blocks with upsampling to increase the image size
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(num_gen_features),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_gen_features, 80, 3, stride=2, padding=1),

            nn.BatchNorm2d(80, 0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(80, 64, 3, stride=2, padding=1),

            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 20, 2, stride=1, padding=1),

            nn.BatchNorm2d(20, 0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(20, 3, 3, stride=1, padding=0),
            
            nn.Tanh())

    def forward(self, x):
        #print('Input = ', x.size())
        out = self.latent_reshape(x)
        #print('Out_linear = ', out.size())
        out = out.view(out.shape[0], num_gen_features, gen_init_image, gen_init_image)
        #print('Reshaping Size = ', out.size())
        img = self.conv_block(out)
        #print(img.size())
        return img
    
# Function to initialize weights as recommended Normal(0, 0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Function for plotting torch normalized images with matplotlib
def prep_image(img):
    ret_img = 0.5 *(1 + torch.permute(img, (1, 2, 0)))
    return ret_img


# Reading in the images in both folders
data_matrix = []
tensor_maker = transforms.ToTensor()

list_images = os.listdir('/home/ethanbrown/data_general/data_jpg/')
list_images = [x for x in list_images if x[0] != '.']

for i in list_images:
   png_img = Image.open('/home/ethanbrown/data_general/data_jpg/' + i)
   img_arr = tensor_maker(png_img)
   data_matrix.append(img_arr)
   png_img.close()

# Doing some wacky stuff to get the images in the correct layout and values
# i.e. between 0 and 1 and then making it a numpy array again so that the whole thign
# can be a batched tensor with N, num_channels, H, W
for i, j in enumerate(data_matrix):
   data_matrix[i] = j.numpy()
data_matrix = np.array(data_matrix)
data_matrix = torch.Tensor(data_matrix)
# Normalizing between -1  and 1 for tanh
transform = nn.Sequential(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
data_matrix_tensor = transform(data_matrix)
print('Data Shape:')
print(data_matrix_tensor.shape)

# Generating the dataloader
mydata = TensorDataset(data_matrix_tensor)
data_loader = DataLoader(mydata, batch_size=batch_size, shuffle=True)

# Loss function
loss_func = torch.nn.BCELoss()

# Initialize generator and discriminator
gen = Generator().to(device)
dis = Discriminator().to(device)

# Initialize weights
gen.apply(weights_init)
dis.apply(weights_init)

# Optimizers
gen_optim = torch.optim.Adam(gen.parameters(), lr=gen_learning_rate, betas=(BETA, 0.999))
dis_optim = torch.optim.Adam(dis.parameters(), lr=dis_learning_rate, betas=(BETA, 0.999))

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, imgs in enumerate(data_loader):

        # Loading in the images
        imgs = imgs[0].to(device)

        # Finding the batch_size
        batch_size = imgs.shape[0]

        # Creating the training zeros and ones
        true = torch.ones(batch_size, 1).to(device)
        false = torch.zeros(batch_size, 1).to(device)

        #######
        # Training the generator first
        gen_optim.zero_grad()
        z = torch.Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_SIZE))).to(device)

        # Using the random noise
        gen_imgs = gen(z)

        # Finding the loss
        gen_loss = loss_func(dis(gen_imgs), true)

        #Backpropagating
        gen_loss.backward()
        gen_optim.step()

        #######
        # Now training the discriminator
        dis_optim.zero_grad()

        # Finding the discriminator loss with real and fake data
        true_loss = loss_func(dis(imgs), true)
        false_loss = loss_func(dis(gen_imgs.detach()), false)
        dis_loss = (true_loss + false_loss) / 2

        # Backpropagating the loss
        dis_loss.backward()
        dis_optim.step()

        # Save the results every so many iterations 
        # Every 10 epochs, print the losses
        if epoch % 5 == 0 and i == 0:
          print("Epoch %d/%d -- Dis loss: %f -- Gen loss: %f" % ((epoch), (NUM_EPOCHS), dis_loss.item(), gen_loss.item()))
          # Save 5x5 rows of images
          save_image(gen_imgs.data[:25], SAVE_PATH + 'EPOCH%d_ITER%d.png' % ((epoch), i), nrow=5, normalize=True)

torch.save(gen.state_dict(), '/home/ethanbrown/code/models/generator_test%d.pt' % TRIAL_NUM)
torch.save(dis.state_dict(), '/home/ethanbrown/code/models/discriminator_test%d.pt' % TRIAL_NUM)