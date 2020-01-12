from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy

# Fonction pour creer le modele Keras
def define_model(activation="", C=1, neuron=1,l_rate=0.1):
    
# definir le modele
     model = Sequential()
     model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(1024, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(11, kernel_initializer='uniform', activation='softmax'))

# compiler le modele
     model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
     return model

#Train
data_train = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-LCred_train.csv", delimiter=",", usecols=range(2048))
x_train = data_train[:,0:2048]
y_train = numpy.loadtxt("AFF11-2/Bd_Revues/train.csv", delimiter=",")

#test
data_test = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-LCred_test.csv", delimiter=",", usecols=range(2048))
x_test = data_test[:,0:2048]
y_test = numpy.loadtxt("AFF11-2/Bd_Revues/test.csv", delimiter=",")

model = KerasClassifier(build_fn=define_model)  
#kernel,gamma,C,class_weight,base_estimator__max_depth,min_samples_split,min_samples_split
#reduce_dim__n_components