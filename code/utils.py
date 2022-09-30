from matplotlib import pyplot as plt
import math
import numpy as np
import os
from PIL import Image
from pathlib import Path
import logging


def resize_images(DATA_PATH_STR, IMAGE_SIZE, SAVE_PATH_STR):
   """
   resize_image
   
   DATA_PATH_str (str): Path that leads to your data folder of choice
   IMAGE_SIZE (int): The new size of the images that you want to use for 
   training the DCNN
   SAVE_PATH (str): Path of a directory that should lead to the folder where you
   want the images saved. If left empty, defaults to creating such a folder in the 
   current DATA_PATH directory

   This function should be modified depending on the file structure of image
   data that you have.

   Returns: A path representation of the folder containing the data for the DCNN modified as needed.
   """

   DATA_PATH = Path(DATA_PATH_STR)

   # Check to make sure that data path exists
   assert DATA_PATH.exists() and DATA_PATH.is_dir(),  f'Path to data directory does not exist! Use the correct path for the data!'

   # If the savepath was not entered, then make one in the current directory s
   if SAVE_PATH_STR == None:
      SAVE_PATH = Path('./saved_data').mkdir(parents=True, exist_ok=True)
   else:
      SAVE_PATH = Path(SAVE_PATH_STR)
      SAVE_PATH.mkdir(parents=True, exist_ok=True)

   if len(os.listdir(SAVE_PATH)) > 15: 
      print(f'Save directory has more than 15 files, and so quitting to avoid overlapping.')
      return SAVE_PATH


   ########
   #
   # The specific of the below function will change
   # depending on the structure of the implementation
   # of your data.
   # This is subject to change for the data provided in 
   # the README.
   #
   ########


   # Then transforming images
   pic_iter = 0

   # Using os.walk to gather information about the images present in the data
   for root, _, files in os.walk(DATA_PATH, topdown=False):
      for name in files:
         # Check to make sure things aren't violated (can change based on makeup of data dir)
         if ('NEXUS' in root and name != 'Small.png') or name[0] == '.' or '.zip' in name:
            continue
         image = Image.open(os.path.join(root, name), mode='r')
         image.resize((IMAGE_SIZE,IMAGE_SIZE), Image.LANCZOS ).convert('RGB').save(SAVE_PATH / f'picnum{pic_iter}.jpg')
         pic_iter = pic_iter + 1
         image.close()
   
   return SAVE_PATH
