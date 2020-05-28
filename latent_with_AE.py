from autoencoder import Autoencoder
from keras.datasets import mnist
import numpy as np

RUN_FOLDER = 'run'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train[:, :, :, np.newaxis].astype('float32') / 255.

AE = Autoencoder(
    input_dim=(28, 28, 1)
    , encoder_conv_filters=[32, 64, 64, 64]
    , encoder_conv_kernel_size=[3, 3, 3, 3]
    , encoder_conv_strides=[1, 2, 2, 1]
    , decoder_conv_t_filters=[64, 64, 32, 1]
    , decoder_conv_t_kernel_size=[3, 3, 3, 3]
    , decoder_conv_t_strides=[1, 2, 2, 1]
    , z_dim=2
)

AE.save(RUN_FOLDER)
AE.encoder.summary()
AE.decoder.summary()

LEARNING_RATE = 0.0005
BATCH_SIZE = 8
INITIAL_EPOCH = 0

AE.compile(LEARNING_RATE)

AE.train(
    x_train[:60000]
    , batch_size=BATCH_SIZE
    , epochs=10
    , run_folder=RUN_FOLDER
    , initial_epoch=INITIAL_EPOCH
)
AE.save_networks(RUN_FOLDER)
