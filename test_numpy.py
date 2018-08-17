#########################################################
# Imports
#########################################################

######################################
# Keras
######################################
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
######################################
# Utils
######################################
from util.extendable_datagen import ImageDataGenerator, random_transform,standardize
import numpy as np

# Run parameters
batch_size = 32
epochs = 10

# Data specific constants
image_size = 128,128

#########################################################
# Network Architecture
#########################################################
inputs = Input((image_size[0], image_size[1], 3)) #RGB


x = Conv2D(6, kernel_size=(5,5), strides=(1, 1), padding='same')(inputs)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(12, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(1, activation='relu')(x)

x = Activation('sigmoid')(x)

model = Model(inputs, x)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])

xt,yt = np.random.uniform(size=(1000,128,128,3)) , np.random.randint(0,2,size=[1000,]).astype('float32')
print("GroundTruth: {}".format(model.evaluate(xt,yt,batch_size=32)))

myDataGen = ImageDataGenerator()
for sshuffle in [False,True]:
    flow_test = myDataGen.flow(xt,yt,batch_size=32,shuffle=sshuffle)
    for multiProc in [False,True]:
        for workers in [1,8]:
            print('Evaluating datagen, shuffle={} , multiprocessing={}, workers={}   - results {}'.format(sshuffle,
                                                            multiProc, workers, 
                                                            model.evaluate_generator(flow_test,workers=workers,use_multiprocessing=multiProc)))
