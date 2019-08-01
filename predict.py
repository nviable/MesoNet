from tqdm import tqdm
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir, exists
from math import floor
import imageio
import face_recognition
from scipy.ndimage.interpolation import zoom, rotate
import tensorflow as tf
import dlib
dlib.DLIB_USE_CUDA = True
from classifiers import MesoInception4, Meso4

from matplotlib import pyplot as plt
from matplotlib import image as pltimg
SHOW_FACES = False

class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
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

class FaceFinder(Video):
    def __init__(self, path, target_size = 256):
        super().__init__(path)
        self.faces = np.zeros((1, target_size, target_size, 3))
        self.coordinates = {} # stores the face (locations center, rotation, length)
        self.last_frame = self.get(0)
        self.frame_shape = self.last_frame.shape[:2]
        face_positions = face_recognition.face_locations(self.last_frame, model="cnn")
        self.last_location = face_positions[0]
        self.target_size = target_size
    
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
    
    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))

    def run(self, resize = 0.5, stop = 0, skipstep = 0, no_face_acceleration_threshold = 3, cut_left = 0, cut_right = -1, use_frameset = False, frameset = []):
        '''
        The core funciton to extract faces from frames
        using previous frame location and downsampling to accelerate the loop
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
        for i in tqdm(finder_frameset, "Extracting faces"):
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
                self.last_location = face_location
                
                # extract face rotation, length and center from landmarks
                landmarks = face_recognition.face_landmarks(frame, [face_location])
                if len(landmarks) > 0:
                    # we assume that there is one and only one landmark group
                    self.coordinates[i] = self.find_coordinates(landmarks[0])

                    patch = self.get_aligned_face(i)
                    self.faces = np.concatenate((self.faces, np.expand_dims(self.resize_patch(patch), axis = 0)), axis = 0)
            else:
                not_found += 1

                if no_face_acc < no_face_acceleration_threshold:
                    # Look for face in full frame
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample = 2, model="cnn")
                else:
                    # Avoid spending to much time on a long scene without faces
                    reduced_frame = zoom(frame, (resize, resize, 1))
                    face_locations = face_recognition.face_locations(reduced_frame, model="cnn")
                    
                if len(face_locations) > 0:
                    print('Face extraction warning : ', i, '- found face in full frame', face_locations)
                    no_face_acc = 0  # reset the no_face_acceleration mode accumulator
                    
                    face_location = self.pop_largest_location(face_locations)
                    
                    # if was found on a reduced frame, upsample location
                    if no_face_acc > no_face_acceleration_threshold:
                        face_location = self.upsample_location(face_location, (0, 0), 1 / resize)
                    
                    self.last_location = face_location
                    
                    # extract face rotation, length and center from landmarks
                    landmarks = face_recognition.face_landmarks(frame, [face_location])
                    if len(landmarks) > 0:
                        self.coordinates[i] = self.find_coordinates(landmarks[0])

                        patch = self.get_aligned_face(i)
                        self.faces = np.concatenate((self.faces, np.expand_dims(self.resize_patch(patch), axis = 0)), axis = 0)
                else:
                    print('Face extraction warning : ',i, '- no face')
                    no_face_acc += 1
                    no_face += 1


        print('Face extraction report of', 'not_found :', not_found)
        print('Face extraction report of', 'no_face :', no_face)
        return 0
    
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

def predict(file_dir, filename, model):
    face_finder = FaceFinder(join(file_dir, filename)) 
    face_finder.run(resize=0.5, skipstep=0)

    data_name = 'f2f'
    weight_path = "./weights"
    # weights.MesoInception-F2F.best_2.h5
    # weight_file = weight_path + '/weights.'+ model_name + '-' + data_name +'.best_2.h5'
    weight_file = join(weight_path, ('weights.' + model + '-' + data_name + '.best.h5')) 
    predictions = []
    m = Meso4(learning_rate = 0.001)
    m.model.summary()
    m.model.load_weights(weight_file)

    if SHOW_FACES:
        pl_num = 1  # variables for grid of faces
        fig = plt.figure()  # variables for grid of faces     
        i = 0
    for face in tqdm(face_finder.faces, desc="Predicting"):
        '''
        > Print a grid of faces
        '''
        if SHOW_FACES:
            if( i%5 == 0 and pl_num <= 4):
                print('adding subplot ', pl_num)
                ax = fig.add_subplot(2,2,pl_num)
                ax.imshow(np.array(face).astype(int), interpolation='nearest')
                pl_num += 1
            if( pl_num == 5 ):
                print('showing plots', pl_num)
                plt.show()
                pl_num += 1
            i += 1
        '''
        < End Print a grid of faces
        '''
        prediction = m.predict(np.expand_dims(face, axis=0))
        predictions.append(prediction[0][0])

    return predictions

if __name__ == "__main__":
    print(predict('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/manipulated_sequences/Face2Face/c23/videos', '035_036.mp4', 'meso4'))
    # print(predict('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/original_sequences/c23/videos', '035.mp4', 'meso4'))

# for e in tqdm(range(n_epoch + 1)):
#     X = []
#     prediction = m.predict(X)
#     predicted[(e * batch_size_test):(e * batch_size_test + prediction.shape[0])] = prediction

# print('predicted mean', np.mean(predicted, axis=0))
# print('class mean :', np.mean(predicted[:split] < 0.5), np.mean(predicted[split:] > 0.5))


## --- Evaluation on video with image aggregation

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])