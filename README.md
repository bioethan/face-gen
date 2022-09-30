## Exploring DnD Portrait Generation with a Variety of Neural Network Architectures

Trial 1 Variables:
------------------------------|
batch_size = 100              |
image_size = 256              |
                              |
gen_init_image = 32           |
num_gen_features = 250        |
                              |
gen_learning_rate =  0.0002   |
dis_learning_rate = 0.0002    |
                              |
# Hyperparameters             |
LATENT_SIZE = 200             |
NUM_EPOCHS = 2500             |
BETA = 0.5                    |
------------------------------|


Generator Structure: 
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


Trial 2 Variables:

TRIAL_NUM = 2
batch_size = 100
image_size = 256

gen_init_image = 8
num_gen_features = 1024

gen_learning_rate =  0.0002
dis_learning_rate = 0.0002

# Hyperparameters
LATENT_SIZE = 200
NUM_EPOCHS = 2500
BETA = 0.5

class Generator(nn.Module):
   def __init__(self):
      super(Generator, self).__init__()

      # Linear layer to take from the initial size to the 
      self.latent_reshape = nn.Linear(LATENT_SIZE, gen_init_image * gen_init_image * num_gen_features)

      # Convolutional blocks with upsampling to increase the image size
      self.conv_block = nn.Sequential(
      #  8x8 input, 12x12 out
      nn.ConvTranspose2d(in_channels=num_gen_features, out_channels=1000, kernel_size=5, stride=1, padding=1),
      nn.BatchNorm2d(1000, 0.8),
      nn.LeakyReLU(0.2),

      # 12x12 in, 25x25 out
      nn.ConvTranspose2d(in_channels=1000, out_channels=500, kernel_size=8, stride=3, padding=1),
      nn.BatchNorm2d(500, 0.8),
      nn.LeakyReLU(0.2),

      # 25x25 in, 96x96 out
      nn.ConvTranspose2d(in_channels=500, out_channels=360, kernel_size=8, stride=4, padding=2),
      nn.BatchNorm2d(360, 0.8),
      nn.LeakyReLU(0.2),

      nn.ConvTranspose2d(in_channels=360, out_channels=3, kernel_size=5, stride=1, padding=4),

      # Activation Function
      nn.Tanh())



