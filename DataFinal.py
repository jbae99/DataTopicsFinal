import imageio as iio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)
import numpy as np
import pandas as pd
import multiprocessing
##import matplotlib.pyplot as plt


## I tested many models, with varying numbers of layers and nodes. 
## I think the relatively large class size that this dataset has (21 Classes) 
## requries more layers in the network. The saving and concatenating of the 
## residual data seems to be the largest contributor to my increased accuracy.

## I did not come up with the concept for this algorithm (which is a simple Xception network)
## I did make changes to it for better use with my data
def makeModel(inputShape, numClass):
    ## Input Layer
    inputs = tf.keras.Input(shape=inputShape)
    # Preprocessing block
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.experimental.preprocessing.RandomFlip()(x)
    x = tf.keras.layers.experimental.preprocessing.RandomRotation(0.25)(x)

    ##standard convolution layer with normalization
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Set aside residual
    previous_block_activation = x  

    ## this loop will use the sizes below to perform separable convolutions 
    ## with increasing node depth. These values are added back together with the
    ##residual from before the loop. 
    for size in [128, 256, 512]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ## one more convolution and normalization block
    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    ##Dropout, Flatten for output, and Output layer configured for multiclass classification
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(21, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

## Mostly defined in a seperate function so as to make it the target of a multiprocess.Process
def trainPredictModel(trainData, valData, testData, model, epochs):
    callbacks = [tf.keras.callbacks.ModelCheckpoint("saveAt{epoch}.keras")]
    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = 'CategoricalCrossentropy',
        metrics = ['accuracy']
    )

    model.fit(
        trainData,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = valData
    )

    model.evaluate(
        testData,
        batch_size = 64
    )

    



def main():
    tf.keras.backend.clear_session()

    ##Constants for convenience  
    imageSize = (200, 200)
    inputShape = imageSize +(3,)
    batchSize = 64
    epochs = 25
    numClass = 21
    imPath = 'd:/Documents/DataTopics/Final/archive/images_train_test_val'

    ##Importing data as data Tensors. Classes are retrieved from directory structure
    ##prefetch increases efficiency on GPU
    train = tf.keras.preprocessing.image_dataset_from_directory(
        imPath + '/train',
        label_mode = 'categorical',
        seed = 420,
        image_size = imageSize, 
        batch_size = batchSize
    )
    train = train.prefetch(tf.data.AUTOTUNE)
    val = tf.keras.preprocessing.image_dataset_from_directory(
        imPath + '/validation',
        label_mode = 'categorical',
        seed = 420,
        image_size = imageSize, 
        batch_size = batchSize
    )
    val = val.prefetch(tf.data.AUTOTUNE)
    test = tf.keras.preprocessing.image_dataset_from_directory(
        imPath + '/test',
        label_mode = 'categorical',
        seed = 420,
        image_size = imageSize,
        batch_size = batchSize
    )
    test = test.prefetch(tf.data.AUTOTUNE)

    model = makeModel(inputShape, numClass)

    ##This is a workaround for the memory issuewith GPUs in Tensorflow
    trainProcess = multiprocessing.Process(target=trainPredictModel(train, val, test, model, epochs))
    trainProcess.start()
    trainProcess.join()
    
    tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()