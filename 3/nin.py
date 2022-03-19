import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import os
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vision = 3

class NiN_model():
    def __init__(self):
      (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
      self.y_train = np_utils.to_categorical(self.y_train, 10)
      self.y_test = np_utils.to_categorical(self.y_test, 10)
      self.x_train, self.x_test = self.color_preprocessing()

      self.lr = 0.1
      self.momentum = 0.9
      self.epochs = 100
      self.batch_size = 100
      self.weight_decay = 0.0001
      
      initial_learning_rate = 0.1
      lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate,
      decay_steps=10000,
      decay_rate=0.1,
      staircase=True) 

      self.optimizer = SGD(learning_rate=lr_schedule, momentum=self.momentum, nesterov=True)
      self.build()

    def color_preprocessing(self):
      self.x_train = self.x_train.astype('float32')
      self.x_test = self.x_test.astype('float32')
      mean = [125.307, 122.95, 113.865]
      std = [62.9932, 62.0887, 66.7048]
      for i in range(3):
        self.x_train[:,:,:,i] = (self.x_train[:,:,:,i] - mean[i]) / std[i]
        self.x_test[:,:,:,i] = (self.x_test[:,:,:,i] - mean[i]) / std[i]
      return self.x_train, self.x_test

    def build(self):
      input_tensor = layers.Input(shape = (32, 32, 3))

      # First mlpConv
      h = layers.Conv2D(192, (5, 5), padding='same')(input_tensor)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(160, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(96, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.MaxPooling2D((3, 3), (2, 2), padding='same')(h)

      h = layers.Dropout(0.5)(h)
      
      # Second mlpConv
      h = layers.Conv2D(192, (5, 5), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(160, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(96, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.MaxPooling2D((3, 3), (2, 2), padding='same')(h)

      h = layers.Dropout(0.5)(h)

      # Third mlpConv
      h = layers.Conv2D(192, (5, 5), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(192, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(96, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.MaxPooling2D((3, 3), (2, 2), padding='same')(h)

      h = layers.Dropout(0.5)(h)

      # Fourth mlpConv
      h = layers.Conv2D(192, (3, 3), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(192, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)
      h = layers.Conv2D(10, (1, 1), padding='same')(h)
      h = layers.Activation('relu')(h)

      # GAP 
      h = layers.GlobalAveragePooling2D()(h)
      y = layers.Activation('softmax')(h)

      self.nin_model = keras.models.Model(input_tensor, y)

      print(self.nin_model.summary())

    def train_model(self):
      self.nin_model.compile(loss="categorical_crossentropy", optimizer=self.optimizer,metrics=['accuracy'])
      # self.history = self.nin_model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.x_test, self.y_test), shuffle=True, verbose=1)
      self.history = self.nin_model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(self.x_test, self.y_test), shuffle=True)
      # datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125, fill_mode='constant', cval=0.)
      # datagen.fit(self.x_train)

      # self.nin_model.fit_generator(datagen.flow(self.x_train, self.y_train, batch_size=self.batch_size), steps_per_epoch=391, epochs=100, validation_data=(self.x_test, self.y_test))

      self.acc = self.history.history["accuracy"]
      self.loss = self.history.history['loss']
      self.val_acc = self.history.history['val_accuracy']

      acc_file = open("acc_" + str(vision) + ".txt", "w")
      count = 0
      for element in self.acc:
          acc_file.write(str(element))
          if count != len(self.acc) - 1:
            acc_file.write("\n")
      acc_file.close()

      loss_file = open("loss_" + str(vision) + ".txt", "w")
      count = 0
      for element in self.loss:
          loss_file.write(str(element))
          if count != len(self.loss) - 1:
            loss_file.write("\n")
      loss_file.close()

      val_acc_file = open("val_acc_" + str(vision) + ".txt", "w")
      count = 0
      for element in self.val_acc:
          val_acc_file.write(str(element))
          if count != len(self.loss) - 1:
            val_acc_file.write("\n")
      val_acc_file.close()

      self.nin_model.save('nin_' + str(vision) + '.h5')

model = NiN_model()

model.train_model() 