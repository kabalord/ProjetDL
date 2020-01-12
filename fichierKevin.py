from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
from keras.layers import Dropout
from keras.constraints import maxnorm

# Fonction pour creer le modele Keras
def define_model(weight_constraint,activation, #hidden_layers,
                 #neurons,optimizer,
                 #learn_rates,momentum,
                 #epochs,batches,
                 dropout_rate): #nb_attributs,dropout_rate=0.0
     # definir le modele
     model = Sequential()
     model.add(Dense(2048, input_dim=2048, kernel_initializer='uniform', activation=activation,
               kernel_constraint=maxnorm(weight_constraint)))
     model.add(Dense(1024, kernel_initializer='uniform', activation=activation))
     model.add(Dense(12, kernel_initializer='uniform', activation='softmax'))
     model.add(Dropout(dropout_rate))
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
#2. Nombre de couches cachées. Tester pour 2, 3, 4 couches.
activations = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
#hidden_layers = [2,3,4]
#param_grid = dict(activation=activations, hidden_layers=hidden_layers)

#3. Nombre de neurones dans les couches cachées.
#4. Algorithme d’optimisation pour le calcul du gradient stochastique (optimizer).
#neurons = [1,5,10,15,20,25,30]
#optimizer = ['SGD']
#param_grid = dict(hidden_layers=hidden_layers, 
#                  neurons=neurons, optimizer=optimizer)

#5. Taux d’apprentissage et Momentum.
#learn_rates = [0.001,0.01,0.1,0.2,0.3]
#momentum = [0.0,0.2,0.4,0.6,0.8,0.9]
#param_grid = dict(learn_rates=learn_rates, momentum=momentum)

#6. Taille du batch et nombre d’epoch
#epochs = numpy.array([50, 100, 150])
#batches = numpy.array([5, 10, 20])
#param_grid = dict(nb_epoch=epochs, batch_size=batches)

#7. Régularisation par Dropout
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint,
                  activation=activations)

#8. Dégradation du taux d’apprentissage

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









