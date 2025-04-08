import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.utils import to_categorical

class Model:
    def __init__(self, X_train):
        self.create_model(X_train)

    def create_model(self, X_train):
        self.model = Sequential()
        self.model.add(InputLayer(input_shape=(X_train.shape[1], )))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        return

    def compile_model(
            self,
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
    ):
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )

    def train_model(self, X_train, y_train, epochs=50, validation_split=0.2, verbose=3):
        print("Training model")
        self.model.fit(
                X_train, 
                y_train, 
                epochs=epochs, 
                validation_split=validation_split, 
                verbose=verbose
        )
        print("Model trained")
