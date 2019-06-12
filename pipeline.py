# -*- coding:utf-8 -*-

#%%
import random
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists

import numpy as np
from math import floor, ceil
from scipy.ndimage.interpolation import zoom, rotate
from matplotlib import pyplot as plt
from matplotlib import image as pltimg
from random import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint

import imageio
import face_recognition

import cv2

# flags
SHOW_FACES = False  # view a grid of faces while they are being extracted
SAVE_OUTPUT = False  # output the predictions as a video with annotated prediction result per frame


## Face extraction

#%%
class Video:
    def __init__(self, path, is_video=True):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg') if is_video else FauxContainer(path)
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']
    
    def init_head(self):
        self.container.set_image_index(0)
    
    def next_frame(self):
        self.container.get_next_data()
    
    def get(self, key):
        return self.container.get_data(key)
    
    def __call__(self, key):
        return self.get(key)
    
    def __len__(self):
        return self.length
    
    def show_frame(self, key):
        self.container.show_frame(key)

#%%
'''
    Imitate imageio.get_reader class functions to make the code think it's reading from a video
'''
class FauxContainer:    
    def __init__(self, path, ext = ['.jpg', '.png']):
        self.path = path
        self.current_index = 0
        self.frames = [imageio.imread(join(path, f)) for f in listdir(path) if isfile(join(path, f)) and ((f[-4:] in ext))]
        self.length = self.get_length()

    def get_next_data(self):
        if( self.current_index == self.length):
            print('Reached end of frames, resetting to 0')
            self.current_index = 0
        self.current_index += 1
        return self.frames[self.current_index-1]
    
    def get_data(self, key):
        if( self.current_index == self.length):
            print('Reached end of frames, resetting to 0')
            key = 0
        return self.frames[key]
    
    def set_image_index(self, idx):
        self.current_index = idx

    def get_length(self):
        return len(self.frames)
    
    def get_meta_data(self):
        return { 'fps': 30 }
    
    def show_frame(self, key):
        plt.imshow(self.get_data(key), interpolation='nearest')
        plt.show()

    
#%%
class FaceFinder(Video):
    def __init__(self, path, load_first_face = True, is_video=True):
        super().__init__(path, is_video)
        self.faces = {}
        self.coordinates = {}  # stores the face (locations center, rotation, length)
        self.last_frame = self.get(0)
        self.frame_shape = self.last_frame.shape[:2]
        self.last_location = (0, 200, 200, 0)
        if (load_first_face):
            face_positions = face_recognition.face_locations(self.last_frame, number_of_times_to_upsample=2)
            if len(face_positions) > 0:
                self.last_location = face_positions[0]
    
    def load_coordinates(self, filename):
        np_coords = np.load(filename)
        self.coordinates = np_coords.item()
    
    def expand_location_zone(self, loc, margin = 0.2):
        ''' Adds a margin around a frame slice '''
        offset = round(margin * (loc[2] - loc[0]))
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)
    
    @staticmethod
    def upsample_location(reduced_location, upsampled_origin, factor):
        ''' Adapt a location to an upsampled image slice '''
        y0, x1, y1, x0 = reduced_location
        Y0 = round(upsampled_origin[0] + y0 * factor)
        X1 = round(upsampled_origin[1] + x1 * factor)
        Y1 = round(upsampled_origin[0] + y1 * factor)
        X0 = round(upsampled_origin[1] + x0 * factor)
        return (Y0, X1, Y1, X0)

    @staticmethod
    def pop_largest_location(location_list):
        max_location = location_list[0]
        max_size = 0
        if len(location_list) > 1:
            for location in location_list:
                size = location[2] - location[0]
                if size > max_size:
                    max_size = size
                    max_location = location
        return max_location
    
    @staticmethod
    def L2(A, B):
        return np.sqrt(np.sum(np.square(A - B)))
    
    def find_coordinates(self, landmark, K = 2.2):
        '''
        We either choose K * distance(eyes, mouth),
        or, if the head is tilted, K * distance(eye 1, eye 2)
        |!| landmarks coordinates are in (x,y) not (y,x)
        '''
        E1 = np.mean(landmark['left_eye'], axis=0)
        E2 = np.mean(landmark['right_eye'], axis=0)
        E = (E1 + E2) / 2
        N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
        B1 = np.mean(landmark['top_lip'], axis=0)
        B2 = np.mean(landmark['bottom_lip'], axis=0)
        B = (B1 + B2) / 2

        C = N
        l1 = self.L2(E1, E2)
        l2 = self.L2(B, E)
        l = max(l1, l2) * K
        if (B[1] == E[1]):
            if (B[0] > E[0]):
                rot = 90
            else:
                rot = -90
        else:
            rot = np.arctan((B[0] - E[0]) / (B[1] - E[1])) / np.pi * 180
        
        return ((floor(C[1]), floor(C[0])), floor(l), rot)
    
    
    def find_faces(self, resize = 0.5, stop = 0, skipstep = 0, no_face_acceleration_threshold = 3, cut_left = 0, cut_right = -1, use_frameset = False, frameset = []):
        '''
        The core function to extract faces from frames
        using previous frame location and downsampling to accelerate the loop.
        '''
        not_found = 0
        no_face = 0
        no_face_acc = 0
        
        # to only deal with a subset of a video, for instance I-frames only
        if (use_frameset):
            finder_frameset = frameset
        else:
            if (stop != 0):
                finder_frameset = range(0, min(self.length, stop), skipstep + 1)
            else:
                finder_frameset = range(0, self.length, skipstep + 1)
        
        # Quick face finder loop
        for i in finder_frameset:
            # Get frame
            frame = self.get(i)
            if (cut_left != 0 or cut_right != -1):
                frame[:, :cut_left] = 0
                frame[:, cut_right:] = 0            
            
            # Find face in the previously found zone
            potential_location = self.expand_location_zone(self.last_location)
            potential_face_patch = frame[potential_location[0]:potential_location[2], potential_location[3]:potential_location[1]]
            potential_face_patch_origin = (potential_location[0], potential_location[3])
    
            reduced_potential_face_patch = zoom(potential_face_patch, (resize, resize, 1))
            reduced_face_locations = face_recognition.face_locations(reduced_potential_face_patch, model = 'cnn')
            
            if len(reduced_face_locations) > 0:
                no_face_acc = 0  # reset the no_face_acceleration mode accumulator

                reduced_face_location = self.pop_largest_location(reduced_face_locations)
                face_location = self.upsample_location(reduced_face_location,
                                                    potential_face_patch_origin,
                                                    1 / resize)
                self.faces[i] = face_location
                self.last_location = face_location
                
                # extract face rotation, length and center from landmarks
                landmarks = face_recognition.face_landmarks(frame, [face_location])
                if len(landmarks) > 0:
                    # we assume that there is one and only one landmark group
                    self.coordinates[i] = self.find_coordinates(landmarks[0])
            else:
                not_found += 1

                if no_face_acc < no_face_acceleration_threshold:
                    # Look for face in full frame
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample = 2)
                else:
                    # Avoid spending to much time on a long scene without faces
                    reduced_frame = zoom(frame, (resize, resize, 1))
                    face_locations = face_recognition.face_locations(reduced_frame)
                    
                if len(face_locations) > 0:
                    print('Face extraction warning : ', i, '- found face in full frame', face_locations)
                    no_face_acc = 0  # reset the no_face_acceleration mode accumulator
                    
                    face_location = self.pop_largest_location(face_locations)
                    
                    # if was found on a reduced frame, upsample location
                    if no_face_acc > no_face_acceleration_threshold:
                        face_location = self.upsample_location(face_location, (0, 0), 1 / resize)
                    
                    self.faces[i] = face_location
                    self.last_location = face_location
                    
                    # extract face rotation, length and center from landmarks
                    landmarks = face_recognition.face_landmarks(frame, [face_location])
                    if len(landmarks) > 0:
                        self.coordinates[i] = self.find_coordinates(landmarks[0])
                else:
                    print('Face extraction warning : ',i, '- no face')
                    no_face_acc += 1
                    no_face += 1

        print('Face extraction report of', 'not_found :', not_found)
        print('Face extraction report of', 'no_face :', no_face)
        return 0
    
    def get_face(self, i):
        ''' Basic unused face extraction without alignment '''
        frame = self.get(i)
        if i in self.faces:
            loc = self.faces[i]
            patch = frame[loc[0]:loc[2], loc[3]:loc[1]]
            return patch
        return frame
    
    @staticmethod
    def get_image_slice(img, y0, y1, x0, x1):
        '''Get values outside the domain of an image'''
        m, n = img.shape[:2]
        padding = max(-y0, y1-m, -x0, x1-n, 0)
        padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return padded_img[(padding + y0):(padding + y1),
                        (padding + x0):(padding + x1)]
    
    def get_aligned_face(self, i, l_factor = 1.3):
        '''
        The second core function that converts the data from self.coordinates into an face image.
        '''
        frame = self.get(i)
        if i in self.coordinates:
            c, l, r = self.coordinates[i]
            l = int(l) * l_factor # fine-tuning the face zoom we really want
            dl_ = floor(np.sqrt(2) * l / 2) # largest zone even when rotated
            patch = self.get_image_slice(frame,
                                    floor(c[0] - dl_),
                                    floor(c[0] + dl_),
                                    floor(c[1] - dl_),
                                    floor(c[1] + dl_))
            rotated_patch = rotate(patch, -r, reshape=False)
            # note : dl_ is the center of the patch of length 2dl_
            return self.get_image_slice(rotated_patch,
                                    floor(dl_-l//2),
                                    floor(dl_+l//2),
                                    floor(dl_-l//2),
                                    floor(dl_+l//2))
        return frame


## Face prediction

class FaceBatchGenerator:
    '''
    Made to deal with framesubsets of video.
    '''
    def __init__(self, face_finder, target_size = 256, cl=-1):
        self.finder = face_finder
        self.target_size = target_size
        self.head = 0
        self.length = int(face_finder.length)
        self.cl = cl

    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))
    
    def next_batch(self, batch_size = 50):
        batch = np.zeros((1, self.target_size, self.target_size, 3))
        # stop = min(self.head + batch_size, self.length)  # seems to be unused
        i = 0
        if SHOW_FACES:
            pl_num = 1  # variables for grid of faces
            fig = plt.figure()  # variables for grid of faces        
        while (i < batch_size) and (self.head < self.length):
            if self.head in self.finder.coordinates:
                patch = self.finder.get_aligned_face(self.head)
                '''
                > Print a grid of faces
                '''
                # print(">> Adding patch of shape: ", np.shape(patch), np.shape(np.expand_dims(self.resize_patch(patch), axis = 0)))
                if( SHOW_FACES and i%5 == 0 and pl_num <= 4):
                    print('adding subplot ', pl_num)
                    ax = fig.add_subplot(2,2,pl_num)
                    ax.imshow(patch, interpolation='nearest')
                    pl_num += 1
                if( SHOW_FACES and pl_num == 5 ):
                    print('showing plots', pl_num)
                    plt.show()
                    pl_num += 1
                '''
                < End Print a grid of faces
                '''
                batch = np.concatenate((batch, np.expand_dims(self.resize_patch(patch), axis = 0)),
                                        axis = 0)
                i += 1
            self.head += 1
        return batch[1:] if self.cl == -1 else batch[1:], [self.cl] * len(batch[1:])


def predict_faces(generator, classifier, batch_size = 50, output_size = 1):
    '''
    Compute predictions for a face batch generator
    '''
    n = len(generator.finder.coordinates.items())
    profile = np.zeros((1, output_size))
    for epoch in range(n // batch_size + 1):
        face_batch = generator.next_batch(batch_size = batch_size)
        prediction = classifier.predict(face_batch)
        if (len(prediction) > 0):
            profile = np.concatenate((profile, prediction))
    return profile[1:]

def data_generator(files, batch_size = 50, ignore_folders=[], frame_cutoff=-1):
    total = len(files)
    i = 0
    while True:
        vid = files[i][0]
        y = files[i][1]
        is_video = files[i][2]
        if i == total-1:
            print("### ran out of data, going back to list HEAD ###")
            i = 0  # start from beginning of the list if we run out of data
        # print(">>>>> Working on vid: ", vid)
        face_finder = FaceFinder(vid, load_first_face = False, is_video=is_video)
        face_finder.find_faces(resize=0.5, skipstep = 0)  # skipstep 0 because we do not want to skip frames in training
        gen = FaceBatchGenerator(face_finder, cl=y)

        while True:
            try: 
                x, y = gen.next_batch(batch_size = batch_size)
                if np.shape(x)[0] == 0:
                    # print("Got 0 lol plz help")
                    break
                # print(">>> Spitting out data of shape: ", np.shape(x), np.shape(y))
                # print(">>> Spitting out data: ", files[i], y)
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

    graph_path = "graphs"
    if not exists(graph_path):
        makedirs(graph_path)
    
    weight_path = "weights"
    if not exists(weight_path):
        makedirs(weight_path)
    
    model_path = "models"
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
    
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=epochs_to_wait_for_improve, verbose=1, restore_best_weights=True)
    checkpoint_callback = ModelCheckpoint(weight_checkpoint_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

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
    

'''
    Complile predictions into a video and annotate each frame with a probability value
'''
def compile_predictions(name, face_finder, predictions):
    img = face_finder.get(0)
    out = cv2.VideoWriter("results/{}.avi".format(name), cv2.VideoWriter_fourcc(*'XVID'), 5, (img.shape[1], img.shape[0]))
    #  messing up the image shape makes it silently break
    for i, p in enumerate(predictions):
        # print('working on frame {} of video {}'.format(i, name))  # extra logging
        img = cv2.putText( face_finder.get(i),'fakeness prob: ' + str(p),
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)
        # plt.imshow(img, interpolation='nearest')  # check whether frames are being annotated correctly
        # plt.show()
        out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    out.release()


def compute_accuracy(classifier, dirname, frame_subsample_count = 30, is_video=True, ignore_folders=[]):
    if is_video:
        '''
        Extraction + Prediction over a video
        '''    
        filenames = [f for f in listdir(dirname) if isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    else:
        '''
        Prediction over a sequence of images (extracted from a video)
        ''' 
        filenames = [f for f in listdir(dirname) if isdir(join(dirname, f)) and (f not in ['processed', 'head', 'head2', 'head3', *ignore_folders])]
    predictions = {}
    
    for count, vid in enumerate(filenames):
        if count == 10:
            break
        print('Dealing with video ', vid)
        
        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face = False, is_video=is_video)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep = 0)  # changed skipstep
        
        print('Predicting ', vid)
        gen = FaceBatchGenerator(face_finder)
        p = predict_faces(gen, classifier)


        prediction = np.mean(p > 0.5)
        decision = '[FAKE]' if prediction>=0.5 else '[REAL]'
        if SAVE_OUTPUT:
            compile_predictions("{}-{}".format(vid, decision), face_finder, p)
        print( 'Predicted video {} to be {} with accuracy of {}'.format(vid, decision, (prediction, p)) )

        if(is_video):
            predictions[vid[:-4]] = (prediction, p)
        else:
            predictions[vid] = (prediction, p)
    return predictions
