import os
import time
import numpy
import pandas as pd
import shutil
import array 

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# step 2. Load Dataset
df = pd.read_csv('accel_x_final_dataset.csv')

dataset = df.values

X = dataset[:,0:12].astype(float) # sensor data
Y = dataset[:,12].astype(int) # labels


# step 3. Define Neural Network Model
def create_model():
    # Define model
    global model
    model = Sequential()
    model.add(Dense(15, input_dim=12, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# step 4. Configure model callbacks including early stopping routine
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
loss_history = LossHistory()
early_stopping = EarlyStopping(monitor='val_acc', patience=20)


# step 5. Assemble classifier and train it
from keras.utils.np_utils import to_categorical

estimator = KerasClassifier(create_model, epochs=200, batch_size=100, verbose=False)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.0003, random_state=5)
Y_test = to_categorical(Y_test)

results = estimator.fit(X_train, Y_train, callbacks=[loss_history, early_stopping], validation_data=(X_test, Y_test))

#step 6. perform 10-fold cross-validation on validation data
kfold = KFold(n_splits=2, shuffle=True, random_state=5)

#collecting latency

##collecting latency

#loops= int(input("no of interation:"))# or 10000 ) 
#interval= float(input("sleep interval(second):")) #or  0.000004 )
#latency = numpy.zeros((loops,1))
loops= 1000 #int(input("no of interation:"))# or 10000 ) 
interval= 0.00040 #float(input("sleep interval(second):")) #or  0.000004 )
latency = numpy.zeros((loops,1))

##2
f_handle=open("raw_output_latency.txt", "a+")
acc_handle=open("raw_output_accuracy.txt", "a+")
f_handle.write("latency coming up : \r\n" )
for i in range(loops):
    t_a=time.time()
    time.sleep(interval)
    t_now = time.time()
    latency[i,0] = t_now-t_a-interval
    f_handle.write("latency : %0.6f\r\n" % latency[i,0])
    cv_results = cross_val_score(estimator, X_test, Y_test, cv=kfold)
    #print("Baseline on test data: %.2f%% (%.2f%%)" % (cv_results.mean()*100, cv_results.std()*100))
    acc_handle.write("Baseline on test data: %.2f%% (%.2f%%)\r\n" % (cv_results.mean()*100, cv_results.std()*100))

f_handle.close()
acc_handle.close()

hist, bin_edges = numpy.histogram(latency, range=(0, latency.max()), bins = 100)
for i in range(hist.size):
     print(bin_edges[i]*1000000," : ",hist[i])


