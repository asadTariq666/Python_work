import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import models, layers
from keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Activation,Dropout

# Loading the dataset
Cifar10=keras.datasets.cifar10 
(X_train,y_train),(X_test,y_test)= Cifar10.load_data()
print('---------------------------------------------------------------------------')
print('Shapes of training and Test sets: ')
print('Training set Descriptive Features: ',X_train.shape)
print('Training set Target Feature: ',y_train.shape)
print('Test set Descriptive Features: ',X_test.shape)
print('Test set Target Feature: ',y_test.shape)

print('---------------------------------------------------------------------------')
class_names =['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
print('class names are: ',class_names)

# One hot Encoding
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
print('---------------------------------------------------------------------------')
# After one hot Encoding
print('Shapes of training and test sets are:')
print((y_train.shape, y_train[0]))
print((y_test.shape, y_test[1]))    


print('---------------------------------------------------------------------------')

# Creating Convolution Neural Netword


# creating an empty sequential model 
model=models.Sequential()
# Adding CNN Layers in the Neural Network with Relu activation function
model.add(layers.Conv2D(64,(3,3),input_shape=(32,32,3),activation='relu'))
model.add(layers.Conv2D(64,(3,3),input_shape=(32,32,3),activation='relu'))
# Max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Second Convolution 
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
# Max pooling layer
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# Third convolutional 
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#Flatten Layer
model.add(layers.Flatten(input_shape=(32,32))) 
# Classification segment 
model.add(layers.Dense(128, activation='relu')) 
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(80, activation='relu')) 

# Adding final output layer to the neural network
model.add(layers.Dense(10, activation='softmax')) 

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#model.summary()

# Training the Convolution Neural Network and evaluating the accuracy. 
X_train2=X_train.reshape(50000,32,32,3)
X_test2=X_test.reshape(10000,32,32,3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model.fit(X_train2,y_train,epochs=40,batch_size=56,verbose=True,validation_data=(X_test2,y_test))

train_loss, training_accuracy = model.evaluate(X_train2, y_train)
test_loss, test_accuracy = model.evaluate(X_test2, y_test)
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print("4. Accuracy of Function 4  'Convolution Neural Network' on Training set:", training_accuracy*100,'%')
print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
print("5. Accuracy of Function 4  'Convolution Neural Network' on Test set:", test_accuracy*100,'%')