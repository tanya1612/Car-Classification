import keras
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
vgg16_model = keras.applications.vgg16.VGG16()

classifier = Sequential()
for layer in vgg16_model.layers:
    classifier.add(layer)
classifier.summary()
for layer in classifier.layers:
    layer.trainable = False

classifier.summary()
classifier.layers.pop()
classifier.add(Dropout(.2))
classifier.add(Dense(2048, activation = 'relu'))
classifier.add(Dense(1024, activation = 'relu'))
classifier.add(Dense(6, activation = 'softmax'))

#Load the new .hdf5 file
file = 'car_9class.hdf5'
classifier.load_weights(file)

#Form test_set and training_set
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

#Testing a Random Image
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
for i in range(1,11):
    img_path = 'test_trained/conver/'+str(i)+'.jpg'   #input image path to predict
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(training_set.class_indices)
    print(result)
    plt.imshow(image.load_img(img_path))
    if result[0][0]>=0.5:
        prediction = 'Convertible'
        print(prediction)
        plt.imshow(image.load_img(img_path))
    elif result[0][1]>=0.5:
        prediction = 'Limousine'
        print(prediction)
        plt.imshow(image.load_img(img_path))
    elif result[0][2]>=0.5:
        prediction = 'Minivan'
        print(prediction)
        plt.imshow(image.load_img(img_path))
    elif result[0][3]>=0.5:
        prediction = 'Pickup'
        print(prediction)
        plt.imshow(image.load_img(img_path))
    elif result[0][4]>=0.5:
        prediction = 'SUV'
        print(prediction)
        plt.imshow(image.load_img(img_path))
    elif result[0][5]>=0.5:
        prediction = 'Sedan'
        print(prediction)
        plt.imshow(image.load_img(img_path))