# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#model = VGG16(weights='imagenet', include_top=False)
vgg16_model = keras.applications.vgg16.VGG16()

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)
model.summary()

for layer in model.layers:
    layer.trainable = False

model.summary()
model.layers.pop()
model.add(Dense(2048, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('car/training_set',
                                                 target_size = (224, 224),
                                                 batch_size = 25,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('car/test_set',
                                            target_size = (224, 224),
                                            batch_size = 25,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 192,
                         epochs = 25,   
                         validation_data = test_set,
                         validation_steps = 48)

file = 'car_6class.hdf5'
model.save_weights(file, overwrite=True)
model.save('test_trained/abhi_newmodel.h5', overwrite=True)