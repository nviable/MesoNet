from tqdm import tqdm
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from forgery_detection.classifiers import MesoInception4
from pipeline import *

batch_size_test = 200


## --- Training

data_gen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        brightness_range=(0.8, 1.2),
        channel_shift_range=30,
        horizontal_flip=True,
        validation_split=0.1)

gen = data_gen.flow_from_directory(
        'database_train',
        target_size=(256, 256),
        batch_size=75,
        class_mode='binary',
        subset='training')

data_gen_test = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

gen_test = data_gen_test.flow_from_directory(
        'database_validation',
        shuffle=False,
        target_size=(256, 256),
        batch_size=batch_size_test,
        class_mode='binary')

##

m = MesoInception4(learning_rate = 0.001)
# m.model.summary()
# m.model.load_weights('weights/MesoInception_DF')

##

m.model.fit_generator(gen,
        steps_per_epoch=2, epochs=2, verbose=1, callbacks=[],
        validation_steps=1,
        max_queue_size=5, workers=4, use_multiprocessing=True)


## --- Evaluation on image dataset
'''
Ugly and unoptimal code but it was fast enough...
'''

n = gen_test.classes.shape[0]
n_epoch = n // batch_size_test
predicted = np.ones((n, 1)) / 2.
split = np.sum(1 - gen_test.classes)  # index of the last fake image + 1

for e in tqdm(range(n_epoch + 1)):
    X, Y = gen_test.next()
    
    prediction = m.predict(X)
    predicted[(e * batch_size_test):(e * batch_size_test + prediction.shape[0])] = prediction

print('predicted mean', np.mean(predicted, axis=0))
print('class mean :', np.mean(predicted[:split] < 0.5), np.mean(predicted[split:] > 0.5))


## --- Evaluation on video with image aggregation

# predictions = compute_accuracy(classifier, 'test_videos')
# for video_name in predictions:
#     print('`{}` video class prediction :'.format(video_name), predictions[video_name][0])