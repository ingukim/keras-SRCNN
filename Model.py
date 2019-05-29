from keras.layers import Input, Activation, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from psnr import PSNR

input_size=(32,32,1)

def model():
    input_img=Input(shape=(input_size))

    SRCNN = Sequential()
    SRCNN = Conv2D( filters=128, kernel_size=(9,9), strides=(1,1), padding='same', kernel_initializer='he_normal')(input_img)
    SRCNN = Activation('relu')(SRCNN)
    SRCNN = Conv2D( filters=64, kernel_size=(1,1), strides=(1,1), padding='same', kernel_initializer='he_normal')(SRCNN)
    SRCNN = Activation('relu')(SRCNN)
    SRCNN = Conv2D( filters=1, kernel_size=(5,5), strides=(1,1), padding='same', kernel_initializer='he_normal')(SRCNN)
    SRCNN = Model(input_img, SRCNN)

    adam=Adam(lr=1e-4)
    SRCNN.compile(optimizer=adam, loss='mse', metrics=[PSNR, 'mean_squared_error'])

    return SRCNN

def test_model():
    input_img=Input(shape=(None,None,1))

    SRCNN = Sequential()
    SRCNN = Conv2D(filters=128, kernel_size=(9, 9), strides=(1, 1), padding='same', kernel_initializer='he_normal')(input_img)
    SRCNN = Activation('relu')(SRCNN)
    SRCNN = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(SRCNN)
    SRCNN = Activation('relu')(SRCNN)
    SRCNN = Conv2D(filters=1, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer='he_normal')(SRCNN)
    SRCNN = Model(input_img, SRCNN)

    return SRCNN

