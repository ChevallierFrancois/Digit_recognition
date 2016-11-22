from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]


model = Sequential()
model.add(Dense(128, input_dim=input_dim, init='lecun_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, input_dim=128, init='lecun_uniform'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, input_dim=128, init='lecun_uniform'))
model.add(Activation('sigmoid'))

# we'll use MSE for the loss and RMSprop as the optimizer
model.compile(loss='mse', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, nb_epoch=15, batch_size=10, verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=2)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "sub.csv")
