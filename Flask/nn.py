import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import cv2
from PIL import Image

model = Sequential()

def train_nn_model():
    (x_train,y_train), (x_test,y_test) = mnist.load_data()

    x_train = np.concatenate((x_train, x_test))
    y_train = np.append(y_train, y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    image_size = x_train.shape[1]
    input_size = image_size * image_size

    x_train = np.reshape(x_train,[-1,input_size])
    x_train = x_train.astype('float32')/255
    
    x_test = np.reshape(x_test,[-1,input_size])
    x_test = x_test.astype('float32')/255

    model.add(Dense(256, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=25, batch_size=128)
    loss, acc = model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest accuracy: %.1f%%" % (100.0 * acc))

    model.save('model')

def test_image():
    trained_model = load_model('model')
    pain = cv2.imread('4/Img_20221107-203147',cv2.IMREAD_GRAYSCALE)
    print(pain)
    res = Image.fromarray(pain)
    res = np.array(res.resize((28,28)))
    res = np.reshape(res,[-1,784])
    res = res.astype('float32')/255
    print(res.shape)
    pred = trained_model.predict(res)

    print(np.argmax(pred, axis=None, out=None))

train_nn_model()
# test_image()