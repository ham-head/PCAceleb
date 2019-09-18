import sys, os, random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import trange
from datagen import NUM_SAMPLES, DataGenerator

BATCH_SIZE = 32
NUM_EPOCHS = 25
PARAM_SIZE = 32
LR = 0.001
TRAIN = False


def plotScores(scores, test_scores, fname, on_top=True):
    plt.clf()
    ax = plt.gca()
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.grid(True)
    plt.plot(scores)
    plt.plot(test_scores)
    plt.xlabel('Epoch')
    plt.ylim([0.0, 0.01])
    loc = ('upper right' if on_top else 'lower right')
    plt.legend(['Train', 'Test'], loc=loc)
    plt.draw()
    plt.savefig(fname)

import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

print("Loading Keras...")

from keras.initializers import RandomUniform
from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.embeddings import Embedding
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l1
from keras.callbacks import TensorBoard, EarlyStopping
from keras.utils import Sequence, plot_model
from keras import backend as K

X = np.expand_dims(np.arange(NUM_SAMPLES + 1), axis=1)

if TRAIN:
    print("Building Model...")
    model = Sequential()
    model.add(Embedding(NUM_SAMPLES + 1, PARAM_SIZE, input_length=1))
    model.add(Flatten(name='pre_encoder'))
    model.add(Reshape((PARAM_SIZE, 1, 1), name='encoder'))
    model.add(Conv2DTranspose(256, 4, strides=(1,2), padding="same", activation="relu"))
    model.add(Conv2DTranspose(256, 4, strides=(1,2), padding="same", activation="relu")) 
    model.add(Conv2DTranspose(256, 4, strides=(1,2), padding="same", activation="relu"))
    model.add(Conv2DTranspose(128, 4, strides=(1,2), padding="same", activation="relu"))  
    model.add(Conv2DTranspose(128, 4, strides=(1,2), padding="same", activation="relu"))     
    model.add(Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid"))
    model.compile(optimizer=Adam(lr=LR), loss='mse')

    print(model.summary())
	
else:
    print("Loading Model...")
    model = load_model('Encoder.h5')

def save_image(x, fname):
	x = x * 255.0
	img = cv.cvtColor(x, cv.COLOR_RGB2BGR)
	cv.imwrite(fname, img)

print("Compiling SubModels...")
func = K.function([model.get_layer('encoder').input, K.learning_phase()],
				  [model.layers[-1].output])
enc_model = Model(inputs=model.input,
                  outputs=model.get_layer('pre_encoder').output)

rand_vecs = np.random.normal(0.0, 1.0, (8, PARAM_SIZE))

def make_rand_faces(rand_vecs, iters):
	x_enc = enc_model.predict(X, batch_size=8)
	
	x_mean = np.mean(x_enc, axis=0)
	x_stds = np.std(x_enc, axis=0)
	x_cov = np.cov((x_enc - x_mean).T)
	e, v = np.linalg.eig(x_cov)

	np.save('means.npy', x_mean)
	np.save('stds.npy', x_stds)
	np.save('evals.npy', e)
	np.save('evecs.npy', v)
	
	e_list = e.tolist()
	e_list.sort(reverse=True)
	plt.clf()
	plt.bar(np.arange(e.shape[0]), e_list, align='center')
	plt.draw()
	plt.savefig('evals.png')
	
	x_vecs = x_mean + np.dot(v, (rand_vecs * e).T).T
	y_faces = func([x_vecs, 0])[0]
	for i in range(y_faces.shape[0]):
		save_image(y_faces[i], 'rand' + str(i) + '.png')
		if i < 5 and (iters % 10) == 0:
			if not os.path.exists('morph' + str(i)):
				os.makedirs('morph' + str(i))
			save_image(y_faces[i], 'morph' + str(i) + '/img' + str(iters) + '.png')

make_rand_faces(rand_vecs, 0)

print("Training...")

datagen = DataGenerator(batch_size=BATCH_SIZE)
callbacks=[TensorBoard()]
train_loss = []

for iters in trange(NUM_EPOCHS):
	history = model.fit_generator(datagen, callbacks=callbacks)

	loss = history.history['loss'][-1]
	train_loss.append(loss)
	print("Loss: " + str(loss))

	plotScores(train_loss, [], 'EncoderScores.png', True)
	
	if iters % 1 == 0:
		model.save('Encoder.h5')
		
		y_faces = model.predict(X[:8], batch_size=8)
		for i in range(y_faces.shape[0]):
			save_image(y_faces[i], 'gt' + str(i) + '.png')
		
		make_rand_faces(rand_vecs, iters)
		
		print("Saved")

print("Done")
