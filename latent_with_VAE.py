from variational_autoencoder import VariationalAutoencoder
from keras.datasets import mnist
import numpy as np

(x_train, _), (_, _) = mnist.load_data()

x_train = x_train[:, :, :, np.newaxis].astype('float32') / 255.0

vae = VariationalAutoencoder(
    input_dim=(28, 28, 1)
    , encoder_conv_filters=[32, 64, 64, 64]
    , encoder_conv_kernel_size=[3, 3, 3, 3]
    , encoder_conv_strides=[1, 2, 2, 1]
    , decoder_conv_t_filters=[64, 64, 32, 1]
    , decoder_conv_t_kernel_size=[3, 3, 3, 3]
    , decoder_conv_t_strides=[1, 2, 2, 1]
    , z_dim=2
)

vae.encoder.summary()
vae.decoder.summary()

LEARNING_RATE = 0.0005
R_LOSS_FACTOR = 1000

BATCH_SIZE = 32
EPOCHS = 15
INITIAL_EPOCH = 0

vae.compile(LEARNING_RATE, R_LOSS_FACTOR)
vae.train(
    x_train
    , batch_size=BATCH_SIZE
    , epochs=EPOCHS
    , initial_epoch=INITIAL_EPOCH
)
vae.save_networks()
