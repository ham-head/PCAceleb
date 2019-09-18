import numpy as np
import pandas as pd
import cv2 as cv
from keras.utils import Sequence

NUM_SAMPLES=50000
BATCH_SIZE=64
PATH = r"D:\celeba-dataset\img_align_celeba\\"

class DataGenerator(Sequence):


    def __init__(self, n_samples=NUM_SAMPLES, batch_size=BATCH_SIZE, dim=(64, 64), n_channels=3, path=PATH, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = np.arange(n_samples+1)
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):

        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        y = self.__data_generation(list_IDs_temp)

        return np.expand_dims(list_IDs_temp, axis=1), y
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            Y[i,] = DataGenerator.preprocess(ID)

        return Y

    @staticmethod
    def preprocess(ID):
        img = cv.imread(PATH + '{:06d}.jpg'.format(ID + 1))
        img = cv.resize(img, (64,78))
        img = img[7:64+7, 0:64]
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0
        return img
        
