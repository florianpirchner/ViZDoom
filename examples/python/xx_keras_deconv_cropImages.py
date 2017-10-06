'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import cv2
import itertools as it
import time
import glob

images_savefileFolder = "/Users/florianpirchner/Work/dev/tensorflow/git/ViZDoom/examples/python/savedImages_deconv"

def crop():
    for filename in glob.glob(images_savefileFolder+'/*.png'):
      img=cv2.imread(filename)
      crop_img = img[0:24, 0:60] # Crop from x, y, w, h -> 100, 200, 300, 400
      # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
      cv2.imwrite(filename, crop_img)

crop()