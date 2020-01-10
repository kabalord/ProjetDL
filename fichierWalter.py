#from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
#from sklearn.model_selection import StratifiedKFold
from csv import reader
import numpy

# Fonction pour creer le modele Keras
def define_model():
     # definir le modele
     model = Sequential()
     #model.add(Dense(82, input_dim=82, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(82, input_dim=82, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(41, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(11, kernel_initializer='uniform', activation='softmax'))
     # compiler le modele
     model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
     return model
 
seed = 4
#filename = 'AFF11-2/AFF11-FCred_train.csv'
#label = 'AFF11-2/train.csv'
dataset = numpy.loadtxt("AFF11-2/AFF11-LCred_train.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

