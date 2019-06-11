#%%
import numpy as np
from pipeline import *
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

IMGWIDTH = 256
learning_rate = 0.001

'''
Copied down Meso4 model
Why copied? Because I don't want to deal with their classes
'''
def theModel():
    optimizer = Adam(lr = learning_rate)

    x = Input(shape = (IMGWIDTH, IMGWIDTH, 3))

    x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = BatchNormalization()(x4)
    x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = Flatten()(x4)
    y = Dropout(0.5)(y)
    y = Dense(16)(y)
    y = LeakyReLU(alpha=0.1)(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation = 'sigmoid')(y)

    model = KerasModel(inputs = x, outputs = y)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    return model


if __name__ == "__main__":
    # (<directory containing videos or folders for sequences>, <real_0/fake_1>, <is_video>)
    dirnames = [
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/original_sequences/c23/videos', 0, True),
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/manipulated_sequences/Face2Face/c23/videos', 1, True)
    ]
    data_split = (.60, .20, .20)

    filenames = []
    batch_size = 40  # 40 frames per batch
    n_epochs = 100
    training_steps_per_epoch = 300  # kinda like epochs within epochs
    training_validation_steps = 100  # same thing for validation
    test_steps = 100
    epochs_to_wait_for_improve = 5

    train_network(theModel(), dirnames, split=data_split, ignore_folders=[], batch_size=batch_size, n_epochs=n_epochs, filenames=filenames, training_steps_per_epoch=training_steps_per_epoch, training_validation_steps=training_validation_steps, test_steps=test_steps, model_name='meso4', data_name='f2f', epochs_to_wait_for_improve = epochs_to_wait_for_improve)
    