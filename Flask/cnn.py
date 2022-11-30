import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from PIL import Image

model = models.Sequential()

def train_cnn_model():
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    
    y_train = Y_train.astype(int)
    x_train = X_train.reshape(-1,28,28,1)
    x_test  = X_test.reshape(-1,28,28,1)
    x_train, x_test = x_train / 255, x_test / 255

    #Using One-hot Encoding
    y_train_ = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(Y_test)

    #Using data augmentation to generate image dataset with varied images
    generator = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=10, zoom_range=0.15)
    tf.random.set_seed(4)
    image_generator = generator.flow(x_train, y_train_, batch_size=60, seed=4)

    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape = (28,28,1)),        
        layers.Conv2D(32, (4, 4), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu'),        
        layers.Conv2D(64, (4, 4), activation = 'relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),                                         
        layers.Conv2D(128, (3, 3), activation = 'relu'),
        layers.BatchNormalization(),
        layers.Flatten(),                        
        layers.Dense(150, activation = tf.nn.relu),   
        layers.Dense(150, activation = tf.nn.relu),      
        layers.Dense(10, activation = tf.nn.softmax) 
    ])

    #when metric stops improving stop the training
    stop_early = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 3, min_delta = 1e-4, restore_best_weights = True)

    #saving the interim model
    saved_model = tf.keras.callbacks.ModelCheckpoint(filepath = 'interimModel/interim_model', save_best_only = True, save_weights_only = True, monitor='val_accuracy', mode='max')
    
    #This function is used to terminate when encountering NaN
    tn = tf.keras.callbacks.TerminateOnNaN()
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.0003, decay_steps = (x_train.shape[0]//60)//4, decay_rate= 0.80, staircase=True)

    # learning rate scheduler
    learning_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    
    #when metric stops improving reduce the learning rate 
    learningPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4, verbose = 3)

    optimum_function = tf.keras.optimizers.Adam(learning_rate=0.0003) 
    lossEntropy = tf.keras.losses.CategoricalCrossentropy()

    model.compile(optimizer=optimum_function,loss=lossEntropy, metrics=['accuracy'])
    history = model.fit(image_generator, epochs=60, validation_data=(x_test, y_test), steps_per_epoch=(x_train.shape[0]//60), callbacks = [learning_scheduler, learningPlateau, saved_model, stop_early, tn])
    model.save('cnnModel')


def test_image():
    trainedModel = models.load_model('cnnModel')
    pain = cv2.imread('/home/frozenfire/MC-Project-1/Flask/1/Img_20221108-175538',cv2.IMREAD_GRAYSCALE)

    pain = cv2.resize(pain,(28,28),interpolation=cv2.INTER_NEAREST)
    image_data = Image.fromarray(pain)
    image_data = np.array(image_data)
    print(type(image_data))

    image_data = image_data.flatten()

    img_fin = image_data

    black_thresh = 128

    nblack =0
    for pixel in image_data:
        if pixel < black_thresh:
            nblack +=1

    n= len(image_data)

    if(nblack / float(n))< 0.5:
        _, thresh = cv2.threshold(pain,128,255,cv2.THRESH_BINARY_INV)
        img_fin = thresh
    else:
        _, thresh = cv2.threshold(pain,128,255,cv2.THRESH_BINARY)
        img_fin = thresh 

    # img_fin = np.array(img_fin.resize(28,28))

    #res = np.reshape(img_fin,[-1,784])

    img_fin = img_fin.astype('float32')/255
    res = img_fin.reshape(-1,28,28,1)
    pred = trainedModel.predict(res)

    print(np.argmax(pred, axis=None, out=None))

# train_cnn_model()
test_image()