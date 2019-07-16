#%%
import numpy as np
from math import ceil
import tensorflow as tf
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
from scipy.ndimage.interpolation import zoom, rotate
import numpy as np
from math import floor, ceil
from random import shuffle
import imageio
from matplotlib import pyplot as plt
from matplotlib import image as pltimg

IMGWIDTH = 256
learning_rate = 0.001

'''
Copied down Meso4 model
Why copied? Because I don't want to deal with their classes
'''
def theModel():
    optimizer = tf.keras.optimizers.Adam(lr = learning_rate)

    x = tf.keras.layers.Input(shape = (IMGWIDTH, IMGWIDTH, 3))

    x1 = tf.keras.layers.Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

    x2 = tf.keras.layers.Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

    x3 = tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

    x4 = tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

    y = tf.keras.layers.Flatten()(x4)
    y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(16)(y)
    y = tf.keras.layers.LeakyReLU(alpha=0.1)(y)
    y = tf.keras.layers.Dropout(0.5)(y)
    y = tf.keras.layers.Dense(1, activation = 'sigmoid')(y)

    model = tf.keras.models.Model(inputs = x, outputs = y)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    return model


class FaceBatchGeneratorStatic:
    '''
    Made to deal with framesubsets of video.
    '''
    def __init__(self, path, target_size = 256, cl=-1, ext = ['.jpg', '.png']):
        self.faces = [imageio.imread(join(path, f)) for f in listdir(path) if isfile(join(path, f)) and ((f[-4:] in ext))]
        self.target_size = target_size
        self.head = 0
        self.length = len(self.faces)
        self.cl = cl

    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))
    
    def next_batch(self, batch_size = 50):
        batch = np.zeros((1, self.target_size, self.target_size, 3))
        # stop = min(self.head + batch_size, self.length)  # seems to be unused
        i = 0
        
        while (i < batch_size) and (self.head < self.length):
            patch = self.faces[self.head]
            batch = np.concatenate((batch, np.expand_dims(self.resize_patch(patch), axis = 0)),
                                    axis = 0)
            i += 1
            self.head += 1
        return batch[1:] if self.cl == -1 else batch[1:], [self.cl] * len(batch[1:])


def data_generator(files, batch_size = 50, ignore_folders=[], frame_cutoff=-1):
    total = len(files)
    i = 0
    max_batches = ceil(frame_cutoff / batch_size)
    while True:
        vid = files[i][0]
        y = files[i][1]
        batches_used = 0
        
        if i == total-1:
            print("### ran out of data, going back to list HEAD ###")
            i = 0  # start from beginning of the list if we run out of data
        gen = FaceBatchGeneratorStatic(vid, cl=y)

        while True:
            try:
                if batches_used == max_batches:
                    break

                x, y = gen.next_batch(batch_size = batch_size)
                batches_used += 1

                if np.shape(x)[0] == 0:
                    break
            except StopIteration as e:
                break
            yield x, y
        i += 1


def print_training(model, history, evaluation):
    print("{}: {}% | {}: {}%".format(model.metrics_names[0], evaluation[0]*100, model.metrics_names[1], evaluation[1]*100))

    # Plot training & validation accuracy values
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("graphs/accuracy.png")

    # Plot training & validation loss values
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig("graphs/loss.png")


def train_network(model, dirnames, split=(.5, .25, .25), ignore_folders=[], batch_size = 40, n_epochs = 5, filenames = [], training_steps_per_epoch = 5, training_validation_steps = 2, test_steps = 5, model_name="meso4", data_name="f2f", epochs_to_wait_for_improve=5, frame_cutoff=-1):
    """Training Function    

    Args:
        model (model): The model to be used for the training
        dirnames (:array:`tuple`:(`str`,int,bool), optional): Names of the directories to add files from. Required
    """

    graph_path = "graphs"
    if not exists(graph_path):
        makedirs(graph_path)
    
    weight_path = "weights"
    if not exists(weight_path):
        makedirs(weight_path)
    
    model_path = "models"
    if not exists(model_path):
        makedirs(model_path)

    model_path = "logs"
    if not exists(model_path):
        makedirs(model_path)

    for dirname, y, is_video in dirnames:            
        if is_video:
            '''
            Extraction + Prediction over a video
            '''    
            filenames.extend([(join(dirname, f), y, is_video) for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))])
        else:
            '''
            Prediction over a sequence of images (extracted from a video)
            ''' 
            filenames.extend([(join(dirname, f), y, is_video) for f in listdir(dirname) if isdir(join(dirname, f)) and (f not in ['processed', 'head', 'head2', 'head3', *ignore_folders])])
    
    shuffle(filenames)  # shuffle file names

    # split data into train, val and test
    total = len(filenames)
    tr_max = floor(total*split[0])
    val_max = tr_max + floor(total*split[1])
    tr_f, val_f, te_f = filenames[:tr_max], filenames[tr_max:val_max], filenames[val_max:]

    if (frame_cutoff > -1):
        # if a frame_cutoff is mentioned, recalculate the steps for each generator to run on all data per epoch
        training_steps_per_epoch = ceil(len(tr_f) * frame_cutoff / batch_size)
        training_validation_steps = ceil(len(val_f) * frame_cutoff / batch_size)
        test_steps = ceil(len(te_f) * frame_cutoff / batch_size)

    train_generator = data_generator(tr_f, batch_size=batch_size, frame_cutoff=frame_cutoff)
    validation_generator = data_generator(val_f, batch_size=batch_size, frame_cutoff=frame_cutoff)
    test_generator = data_generator(te_f, batch_size=batch_size, frame_cutoff=frame_cutoff)
    weight_checkpoint_file = weight_path + '/weights.'+ model_name + '-' + data_name +'.best.h5'
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve, verbose=1)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(weight_checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    if(exists(weight_checkpoint_file)):
        model.load_weights(weight_checkpoint_file)

    history = model.fit_generator(train_generator, steps_per_epoch=training_steps_per_epoch, verbose=1, epochs=n_epochs, validation_data=validation_generator, validation_steps=training_validation_steps, use_multiprocessing=True, callbacks=[early_stopping_callback, checkpoint_callback])

    evaluation = model.evaluate_generator(test_generator, steps=test_steps)

    print_training(model, history, evaluation)

    model_json = model.to_json()
    with open(model_path + "/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()
        
    model.save_weights(weight_path + '/'+ model_name + '-' + data_name +'_retrained.h5')

    training_settings = {
        'model_name' : model_name,
        'data_name' : data_name,
        'dirnames' : dirnames,
        'data_split' : split,
        'filenames' : filenames,
        'batch_size' : batch_size,
        'n_epochs' : n_epochs,
        'training_steps_per_epoch' : training_steps_per_epoch,
        'training_validation_steps' : training_validation_steps,
        'test_steps' : test_steps,
        'epochs_to_wait_for_improve' : epochs_to_wait_for_improve,
        'frame_cutoff' : frame_cutoff
    }

    with open(model_path + "/" + model_name + ".settings.json", "w") as json_file:
        json_file.write(training_settings)
    json_file.close()

if __name__ == "__main__":
    # (<directory containing videos or folders for sequences>, <real_0/fake_1>, <is_video>)
    dirnames = [
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/original_sequences/extracted/videos', 0, False),
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/manipulated_sequences/Face2Face/extracted/videos', 1, False)
    ]
    data_split = (.60, .20, .20)

    batch_size = 40  # 40 frames per batch
    n_epochs = 100
    training_steps_per_epoch = 300  # kinda like epochs within epochs
    training_validation_steps = 100  # same thing for validation
    test_steps = 100
    epochs_to_wait_for_improve = ceil(n_epochs * 0.2)
    frame_cutoff = 80  # if added, ignores **_steps_** variables
    
    

    
    ''' To run code over all the files per epoch
        Hence if we have 1000 files for each class
        1,000 * 2       = 2,000 files
        Tr, Te, Va      = 1200, 400, 400
        Tr = 1,200 * 80 = 96,000 frames
            96,000 / 40 = 2400 batches
        Te = 400 * 80   = 32,000 frames
            32,000 / 40 = 800 batches
        Va = 400 * 80   = 32,000 frames
            32,000 / 40 = 800 batches
        steps_per_epoch = ceil(filenames * frame_cutoff / batch_size)
    '''

    train_network(theModel(), dirnames, split=data_split, ignore_folders=[], batch_size=batch_size, n_epochs=n_epochs, training_steps_per_epoch=training_steps_per_epoch, training_validation_steps=training_validation_steps, test_steps=test_steps, model_name='meso4', data_name='f2f', epochs_to_wait_for_improve = epochs_to_wait_for_improve, frame_cutoff=frame_cutoff)
    