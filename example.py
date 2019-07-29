#%%
import numpy as np
from classifiers import *
from pipeline import *

from keras.preprocessing.image import ImageDataGenerator

#%%
# 1 - Load the model and its pretrained weights
classifier = Meso4()
'''
classifier.load('weights/Meso4_DF')

# 2 - Minimial image generator
# We did use it to read and compute the prediction by batchs on test videos
# but do as you please, the models were trained on 256x256 images in [0,1]^(n*n)
dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary',
        subset='training')

# 3 - Predict
X, y = generator.next()
print('Predicted :', classifier.predict(X), '\nReal class :', y)
'''
# 4 - Prediction for a video dataset

# classifier.load('weights/Meso4_F2F')
classifier.load('weights/weights.meso4-f2f.best.h5')

#%%
# /home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/dataset/splits/test.json
# /home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/manipulated_sequences/Face2Face/c23/videos
# 
predictions = compute_accuracy(classifier, '/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media', is_video=True, frame_subsample_count=10, json_filenames='/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/dataset/splits/test.json')

# predictions = compute_accuracy(classifier, '/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/images/manipulated_sequences/Face2Face/raw/images', is_video=False, frame_subsample_count=10)

# predictions_real = compute_accuracy(classifier, '/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/images/original_sequences/raw/images', is_video=False, frame_subsample_count=10)

for video_name in predictions:
    print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])