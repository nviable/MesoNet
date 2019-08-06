from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
import tensorflow as tf
import json
from classifiers import MesoInception4
# from pipeline import *

def json_to_pandas(json_file_path, dir0, dir1):
        d = { 'x_col': [], 'y_col': [] }
        with open(json_file_path, 'r') as jf:
                datastore = json.load(jf)
                for pair in datastore:
                        a,b = pair
                        
                        dir_a = join(dir0, a + '.mp4')
                        a_images = [join(dir_a, f) for f in sorted(listdir(dir_a))]
                        a_labels = ['real'] * len(a_images)
                        
                        dir_b = join(dir0, b + '.mp4')
                        b_images = [join(dir_b, f) for f in sorted(listdir(dir_b))]
                        b_labels = ['real'] * len(b_images)

                        dir_c = join(dir1, a + '_' + b + '.mp4')
                        c_images = [join(dir_c, f) for f in sorted(listdir(dir_c))]
                        c_labels = ['fake'] * len(c_images)

                        d['x_col'].extend(a_images)
                        d['x_col'].extend(b_images)
                        d['x_col'].extend(c_images)

                        d['y_col'].extend(a_labels)
                        d['y_col'].extend(b_labels)
                        d['y_col'].extend(c_labels)

                        # d['x_col'].append(dir0 + '/' + a + '.mp4')
                        # d['y_col'].append("real")

                        # d['x_col'].append(dir0 + '/' + b + '.mp4')
                        # d['y_col'].append("real")

                        # d['x_col'].append(dir1 + '/' + a + '_' + b + '.mp4')
                        # d['y_col'].append("fake")
        return pd.DataFrame(data=d)
                

if __name__ == "__main__":
        dir0 = '/home/js8365/dataset-deepfakes/FaceForensics/extracted/real'
        dir1 = '/home/js8365/dataset-deepfakes/FaceForensics/extracted/fake'
        train_data_pd = json_to_pandas(
                '/home/js8365/dataset-deepfakes/FaceForensics/dataset/splits/train.json',
                dir0, dir1)
        
        val_data_pd = json_to_pandas(
                '/home/js8365/dataset-deepfakes/FaceForensics/dataset/splits/val.json',
                dir0, dir1)

        test_data_pd = json_to_pandas(
                '/home/js8365/dataset-deepfakes/FaceForensics/dataset/splits/test.json',
                dir0, dir1)

        batch_size_test = 200
        batch_size_train = 75
        batch_size_val = 75
        ## --- Training

        data_gen = ImageDataGenerator(
                rescale=1./255,
                zoom_range=0.2,
                rotation_range=15,
                brightness_range=(0.8, 1.2),
                channel_shift_range=30,
                horizontal_flip=True
                )
        
        gen = data_gen.flow_from_dataframe(
                train_data_pd,
                x_col='x_col',
                y_col='y_col',
                directory=None,
                shuffle=True,
                target_size=(256, 256),
                batch_size=batch_size_train,
                class_mode='binary'
                )
        # Found 548565 validated image filenames belonging to 2 classes.

        data_gen_val = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True)

        gen_val = data_gen_val.flow_from_dataframe(
                val_data_pd,
                x_col='x_col',
                y_col='y_col',
                directory=None,
                shuffle=True,
                target_size=(256, 256),
                batch_size=batch_size_val,
                class_mode='binary'
                )
        # Found 103125 validated image filenames belonging to 2 classes.

        data_gen_test = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True)

        gen_test = data_gen_test.flow_from_dataframe(
                test_data_pd,
                x_col='x_col',
                y_col='y_col',
                directory=None,
                shuffle=False,
                target_size=(256, 256),
                batch_size=batch_size_test,
                class_mode='binary')
        # Found 112098 validated image filenames belonging to 2 classes.

        train_steps = len(train_data_pd.index) // batch_size_train
        val_steps = len(val_data_pd.index) // batch_size_val
        test_steps = len(test_data_pd) // batch_size_test
        epochs_to_wait_for_improve = 5
        model_name = 'MesoInception' 
        data_name = 'F2F'
        weight_path = "./weights"
        
        print("{} Train Steps | {} Val Steps | {} Test Steps | {} Epochs to wait | {} Model Name".format(
                train_steps, val_steps, test_steps, epochs_to_wait_for_improve, model_name))

        ##
        m = MesoInception4(learning_rate = 0.001)
        m.model.summary()
        m.model.load_weights('weights/MesoInception_F2F')

        ##
        weight_checkpoint_file = weight_path + '/weights.'+ model_name + '-' + data_name +'.best_2.h5'
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve, verbose=1)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(weight_checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        m.model.fit_generator(gen,
                steps_per_epoch=train_steps, epochs=100, verbose=1, callbacks=[checkpoint_callback, early_stopping_callback],
                validation_steps=val_steps, validation_data=gen_val,
                max_queue_size=5, workers=4, use_multiprocessing=True)

        m.model.save_weights(join(weight_path, model_name + '-' + data_name + '_retrained'))
        ## --- Evaluation on image dataset

        '''
        Ugly and unoptimal code but it was fast enough...
        '''

        gen_classes = np.array(gen_test.classes)
        n = gen_classes.shape[0]
        n_epoch = n // batch_size_test
        predicted = np.ones((n, 1)) / 2.
        split = np.sum(1 - gen_classes)  # index of the last fake image + 1

        for e in tqdm(range(n_epoch + 1)):
                X, Y = gen_test.next()

                prediction = m.predict(X)
                predicted[(e * batch_size_test):(e * batch_size_test + prediction.shape[0])] = prediction

        print('predicted mean', np.mean(predicted, axis=0))
        print('class mean :', np.mean(predicted[:split] < 0.5), np.mean(predicted[split:] > 0.5))