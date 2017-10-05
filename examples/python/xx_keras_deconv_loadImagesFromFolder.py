'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import cv2
from scipy.stats import norm
import skimage.color, skimage.transform, skimage.io
from random import sample, randint, random
import itertools as it
import time
import glob

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


images_savefile = "/Users/florianpirchner/work/tensorflow/git/ViZDoom/examples/python/savedImages_deconv/defend_the_line_"
images_savefileFolder = "/Users/florianpirchner/work/tensorflow/git/ViZDoom/examples/python/savedImages_deconv"
model_savefile = "/Users/florianpirchner/work/tensorflow/git/ViZDoom/examples/python/savedModels_deconv/model_weights.h5"
save_model = True
load_model = True
skip_learning = False
collectImagesFromFolder = True

# input image dimensions
img_rows, img_cols, img_chns = 24, 60, 3
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 250

#config_file_path = "../../scenarios/simpler_basic.cfg"
config_file_path = "../../scenarios/defend_the_line.cfg"
# number of images from game
sample_size = 20000

if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 100
intermediate_dim = 128
epsilon_std = 1.0
epochs = 500

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 12 * 30, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 12, 30)
else:
    output_shape = (batch_size, 12, 30, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, 26, 61, 3)
else:
    output_shape = (batch_size, 26, 61, 3, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


y = CustomVariationalLayer()([x, x_decoded_mean_squash])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()


def collectFrames():
    tmpImages = []
    for filename in glob.glob(images_savefileFolder+'/*.png'):
      img=cv2.imread(filename)
      tmpImages.append(img)

    print("found images: ", len(tmpImages))
    return np.asarray(tmpImages)


# train the VAE on MNIST digits
#(x_train, _), (x_test, y_test) = mnist.load_data()

images = collectFrames()
np.random.shuffle(images)

if load_model:
  print("--- loading model")
  vae.load_weights(model_savefile, by_name=True)

valData = images[:250, :, :, :]

if not skip_learning:
  #print("shape valData ", valData.shape)
  vae.fit(images,
          shuffle=True,
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(valData, None))

  # save the model
  vae.save_weights(model_savefile)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(valData, batch_size=batch_size)
##plt.figure(figsize=(6, 6))
##plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
##plt.colorbar()
##plt.show()

# build a digit generator that can sample from the learned distribution
#decoder_input = Input(shape=(latent_dim,))
#_hid_decoded = decoder_hid(decoder_input)
#_up_decoded = decoder_upsample(_hid_decoded)
#_reshape_decoded = decoder_reshape(_up_decoded)
#_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
#_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
#_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
#_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
#generator = Model(decoder_input, _x_decoded_mean_squash)

# display a 2D manifold of the digits
valImage = images[0:1, :, :, :]
print(valImage.shape)
decoded_sample = vae.predict(valImage, 2)
decoded_sample = decoded_sample[0, :, :, :]

print("showing")
cv2.imshow("valImage", valImage[0, :, :, :])
cv2.imshow("decoded_sample", decoded_sample)
cv2.waitKey(0)
##plt.figure(figsize=(10, 10))
##plt.imshow(figure, cmap='Greys_r')
##plt.show()