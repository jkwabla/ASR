import json
import numpy as np
from tensorflow import keras
from keras import layers, models, optimizers, losses, metrics
from sklearn.model_selection import train_test_split



DATA_PATH  =  "data.json"
LEARNING_RATE = 0.0001
BATCH_SIZE =  32
EPOCHS =  40
SAVED_MODEL_PATH = "model.h5"



def load_dataset(data_path):

    with open (data_path, "r") as fp:
            data =  json.load(fp)

    #extracting input & targets    

    x= np.array(data["MFCCs"])
    y= np.array(data["labels"])

    return x,y

        

def get_data_splits(data_path,test_size=0.1,test_validation=0.1):
    
    #load datasets
    x,y  = load_dataset(data_path)

    #create train / validate and test  data splits 

    x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=test_size,)
    x_train,x_validation,y_train,y_validation =  train_test_split(x_train,y_train,test_size=test_validation)


    #convert input from 2dimentional data to 3 dimentional data 
    
    x_train = x_train[...,np.newaxis]
    x_validation =  x_validation[...,np.newaxis]
    x_test = x_test[...,np.newaxis]

    return x_train,x_validation,x_test,y_train,y_validation,y_test


def build_model(input_shape, learning_rate):
      
      #starting from the hightest level
      #build the network - tensorflow & keras 

      model = keras.Sequential()
      
      #CNN

      #conv layer 1

      model.add(keras.layers.Conv2D(64,(3,3),activation="relu",input_shape=input_shape,kernel_regularizer=keras.regularizers.l2(0.001)))     
      model.add(keras.layers.BatchNormalization())   
      model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))

      #conv layer 2

      model.add(keras.layers.Conv2D(32,(3,3),activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))     
      model.add(keras.layers.BatchNormalization())   
      model.add(keras.layers.MaxPool2D((3,3),strides=(2,2),padding="same"))

      #conv layer 3

      model.add(keras.layers.Conv2D(32,(2,2),activation="relu",input_shape=input_shape,kernel_regularizer=keras.regularizers.l2(0.001)))     
      model.add(keras.layers.BatchNormalization())   
      model.add(keras.layers.MaxPool2D((2,2),strides=(2,2),padding="same"))

    
      #flatten the output fred into a dense layer 

      model.add(keras.layers.Flatten())
      model.add(keras.layers.Dense(64,activation="relu"))
      model.add(keras.layers.Dropout(0.3))



def main():
    
    #Load train / validate and test  data splits 

        x_train , x_validation ,x_test , y_train , y_validation , y_test  =  get_data_splits(DATA_PATH)


    #Build the CNN model 

        input_shape = (x_train.shape[1] , x_validation.shape[2] ,x_test.shape[3])
        model  =  build_model(input_shape,LEARNING_RATE)

    #Training the model 

        model.fit(x_train,y_train, epochs = EPOCHS, batch_size=BATCH_SIZE,validation_data=(x_validation,y_validation))

    #Evaluate the Model 
        test_error, test_accuracy = model.evaluate(x_test,y_test)
        print(f"Test Error: {test_error}, test_accuracy: {test_accuracy}")

    # save the model  

        model.save(SAVED_MODEL_PATH)