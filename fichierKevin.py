from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# Fonction pour creer le modele Keras
def define_model(activation="", C=1, neuron=1,l_rate=0.1):
     # definir le modele
     model = Sequential()
     model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(1024, kernel_initializer='uniform', activation='relu'))
     model.add(Dense(12, kernel_initializer='uniform', activation='softmax'))
     # compiler le modele
     model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
     return model

#Train
data_train = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-GCred_train.csv", delimiter=",", usecols=range(2048))
x_train = data_train[:,0:2048]
y_train = numpy.loadtxt("AFF11-2/Bd_Revues/train.csv", delimiter=",")
#test
data_test = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-GCred_test.csv", delimiter=",", usecols=range(2048))
x_test = data_test[:,0:2048]
y_test = numpy.loadtxt("AFF11-2/Bd_Revues/test.csv", delimiter=",")

model = KerasClassifier(build_fn=define_model)
#kernel,gamma,C,class_weight,base_estimator__max_depth,min_samples_split,min_samples_split
#reduce_dim__n_components

#1. Les différentes fonctions d’activation pour les couches cachées (la fonction
#d’activation pour la couche de sortie reste toujours softmax)
#activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
#param_grid = dict(activation=activations)

#2 Nombre de couches cachées. Tester pour 2, 3, 4 couches.
#neuron = [12]
#param_grid = dict(neuron=neuron)

#3 Nombre de neurones dans les couches cachées.
#couche = [2,3,4]
#param_grid = dict(C=couche)

#4. Algorithme d’optimisation pour le calcul du gradient stochastique (optimizer).
#optimizer = ['SGD']
#param_grid = dict(optimizer=optimizer)

#5. Taux d’apprentissage et Momentum.
#l_rate = [0.2]
#param_grid = dict(l_rate=l_rate)

#6. Taille du batch et nombre d’epoch
epochs = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])
param_grid = dict(nb_epoch=epochs, batch_size=batches)

grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train,validation_data=(x_test, y_test))

# afficher les resultats
print("Best: %f with %s" % (grid_result.best_score_, grid_result.best_params_))
# afficher les resultats detailles
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("mean (+/- std) = %f (%f) with: %r" % (mean, stdev, param))




#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#print("total time:",time()-start)

#model = KerasClassifier(build_fn=define_model, epochs=300, batch_size=10, verbose=0)
#neurons=[]
#activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
#grid_param = dict(activation=activations)
#grid = GridSearchCV(estimator=model, param_grid=grid_param, n_jobs=1, cv=3)
#grid_result = grid.fit(x_train, y_train, validation_data=(x_test, y_test))
#print("Best : %f with %s" % (grid_result.best_score_, grid_result.best_params_))









