import tensorflow 
from tensorflow.keras.layers import Input, MaxPool2D, GlobalAvgPool2D, Dense, BatchNormalization, Conv2D
import numpy as np
from tensorflow.keras import Model

def functional_model():
    Input = tensorflow.keras.layers.Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(Input)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    Output = Dense(10, activation='softmax')(x)

    model = Model(Input, Output)

    return model

(train_image, train_label), (test_image, test_label) = tensorflow.keras.datasets.mnist.load_data()

train_image = train_image.astype('float32') / 255.0
test_image = test_image.astype('float32') / 255.0

train_image = np.expand_dims(train_image, axis=-1)
test_image = np.expand_dims(test_image, axis=-1)

model = functional_model()

model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_image, train_label, batch_size=64, epochs=3, validation_split=0.15)
model.evaluate(test_image, test_label, batch_size=64)