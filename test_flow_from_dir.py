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
num_classes = 5

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
x = Dense(num_classes, activation='relu')(x)

x = Activation('softmax')(x)

model = Model(inputs, x)
opt = optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['acc'])
model.summary()

datagen = ImageDataGenerator()
datagen.set_pipeline([random_transform,standardize])
xt,yt=next(iter(datagen.flow_from_directory('./hdrvdp',batch_size=1000,target_size=(128,128),shuffle=True)))
print("GroundTruth: {}".format(model.evaluate(xt,yt,128)))

myDataGen = ImageDataGenerator()
for sshuffle in [False,True]:
    flow_test = myDataGen.flow_from_directory('./hdrvdp',batch_size=32,target_size=(128,128),shuffle=sshuffle)
    for multiProc in [False,True]:
        for workers in [1,8]:
            print('Evaluating datagen, shuffle={} , multiprocessing={}, workers={}   - results {}'.format(sshuffle,
                                                multiProc, workers, 
                                                model.evaluate_generator(flow_test,workers=workers,use_multiprocessing=multiProc)))

