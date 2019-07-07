import numpy as np
from pipeline import *


if __name__ == "__main__":
    # (<directory containing videos or folders for sequences>, <real_0/fake_1>, <is_video>)
    dirnames = [
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/original_sequences/c23/videos', 0, True),
        ('/home/js8365/data/Sandbox/dataset-deepfakes/FaceForensics/media/manipulated_sequences/Face2Face/c23/videos', 1, True)
    ]
    
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
    filenames = []
    ignore_folders = []

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
    
    face_savinator(filenames)