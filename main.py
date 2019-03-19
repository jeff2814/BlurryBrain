import tensorflow as tf
import numpy as np
import struct
import matplotlib.pyplot as plt

# Get and format training images
with open('train-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    x_train = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    x_train = x_train.reshape((size, nrows, ncols))
    
# Get training labels
with open ('train-labels-idx1-ubyte', 'rb') as g:
    magic, size = struct.unpack(">II", g.read(8))
    y_train = np.fromfile(g, dtype=np.dtype(np.uint8).newbyteorder('>'))


# Get and format testing images
with open('t10k-images-idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    x_test = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    x_test = x_test.reshape((size, nrows, ncols))
    
# Get testing labels
with open ('t10k-labels-idx1-ubyte', 'rb') as g:
    magic, size = struct.unpack(">II", g.read(8))
    y_test = np.fromfile(g, dtype=np.dtype(np.uint8).newbyteorder('>'))
    
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

