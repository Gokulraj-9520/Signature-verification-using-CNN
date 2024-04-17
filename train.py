import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import easygui 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
train_set = train_datagen.flow_from_directory(
        './training',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 ,strides=2))
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 ,strides=2))
cnn.add(tf.keras.layers.Flatten()) 
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
epoch=50
history=cnn.fit(x = train_set, epochs = epoch,validation_data=train_set,verbose=1,callbacks=[early_stopping])

loss,acc=cnn.evaluate(train_set)
print(loss,acc)
cnn.save("model.h5")

image1=easygui.fileopenbox()
test_image=image.load_img(image1,target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
print(result)
if result[0][0]==1:
    print("Genuine")
else:
    print("Forged")