    from __future__ import print_function

    import cv2
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, concatenate
    from keras.layers.core import Dropout
    from keras.optimizers import *
    from keras.constraints import maxnorm
    from keras.callbacks import ModelCheckpoint, LearningRateScheduler
    from keras import backend as K
    from keras.utils.vis_utils import plot_model
    from ImageDataGenerator import ImageDataGenerator

    from data import load_train_data, load_test_data

    img_rows = 64
    img_cols = 96

    smooth = 1. 


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. / (dice_coef(y_true, y_pred) + smooth)

def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(inputs)
    conv1 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(128, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(pool1)
    conv2 = Convolution2D(128, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(256, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(pool2)
    conv3 = Convolution2D(256, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(512, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(pool3)
    conv4 = Convolution2D(512, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(1024, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = Convolution2D(1024, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv5)
    conv5 = Dropout(0.5)(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv05 = Convolution2D(2048, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(pool5)
    conv05 = Dropout(0.5)(conv05)
    conv05 = Convolution2D(2048, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv05)
    conv05 = Dropout(0.5)(conv05)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv05), conv5])
    conv6 = Convolution2D(1024, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(up6)
    conv6 = Dropout(0.5)(conv6)
    conv6 = Convolution2D(1024, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv6)
    conv6 = Dropout(0.5)(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv4])
    conv7 = Convolution2D(512, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(up7)
    conv7 = Convolution2D(512, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv3])
    conv8 = Convolution2D(256, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(up8)
    conv8 = Convolution2D(256, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv2])
    conv9 = Convolution2D(128, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(up9)
    conv9 = Convolution2D(128, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv1])
    conv09 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(up10)
    conv09 = Convolution2D(64, (3, 3), activation='relu', kernel_initializer='lecun_uniform', kernel_constraint=maxnorm(3), padding='same')(conv09)

    conv10 = Convolution2D(1, (1, 1), activation='sigmoid')(conv09)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adadelta(lr=0.1, rho=0.95, epsilon=1e-08), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0]  = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_AREA)
    return imgs_p


def swap_axes(imgs):
    return np.swapaxes(np.swapaxes(imgs, 1, 2), 2, 3)


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    datagen = ImageDataGenerator(
            rotation_range=5,
            vertical_flip=True,
            horizontal_flip=True,
            )
    model.fit_generator(datagen.flow(swap_axes(imgs_train), imgs_mask_train, batch_size=32, shuffle=True),
        steps_per_epoch=len(imgs_train), epochs=2, verbose=1, callbacks=[model_checkpoint])


    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(swap_axes(imgs_test[:10]),  verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
