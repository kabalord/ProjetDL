from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

seed = 7
numpy.random.seed(seed)
#Train
x_train = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-2/AFF11-GCred_train")
y_train = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-2/train")
#test
x_test = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-2/AFF11-GCred_test")
y_test = numpy.loadtxt("AFF11-2/Bd_Revues/AFF11-2/test")
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

nb_classes = numpy.unique(y_train)

# Fonction pour creer le modele Keras
def define_model(activation, hidden_layers, neurons, optimizer, learn_rates, momentum,  weight_constraint, dropout_rate ):
     # definir le modele
     model = Sequential()
     # create model
     model.add(Dense(x_train.shape[1], input_dim=x_train.shape[1], kernel_initializer='uniform', activation=activation))
               #kernel_constraint=maxnorm(weight_constraint)))
     model.add(Dense(int(x_train.shape[1]/2), kernel_initializer='uniform', activation=activation))
     model.add(Dense(len(nb_classes), kernel_initializer='uniform', activation='softmax'))
     #model.add(Dropout(dropout_rate))
     # compiler le modele
     model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
     return model

# charger la base mnist est préparer les données
# definir le modèle keras
# compiler le modele

# entrainer le modèle sur l'ensemble train
model = KerasClassifier(build_fn=define_model,epochs=30,batch_size=10)


#Fonctions d’activation & Nombre de couches cachées
activation = ['sigmoid']
hidden_layers = [2]

#Algorithme d’optimisation & Nombre de neurones
neurons = [15]
optimizer = ['Adadelta']

#Taux d’apprentissage et Momentum.
learn_rates = [0.01]
momentum = [0.6]

#Régularisation par Dropout
weight_constraint = [1]
dropout_rate = [0.5]


#Entrainement
param_grid = dict(activation=activation, weight_constraint=weight_constraint, dropout_rate=dropout_rate
                  ,hidden_layers=hidden_layers, neurons=neurons, optimizer=optimizer, learn_rates=learn_rates, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=10)


# afficher les resultats
print("Best: %f with %s" % (grid_result.best_score_, grid_result.best_params_))
# afficher les resultats detailles
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("mean (+/- std) = %f (%f) with: %r" % (mean, stdev, param))


predictions = grid.predict(x_test)
labelNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

#instanciation de l'historique de la grille
H=grid_result.best_estimator_.model.history.history

print(classification_report(y_test, predictions, target_names=labelNames))
f = open("modele_base1.txt", "a")
f.write("classification report") 
f.write(str(classification_report(y_test, predictions, target_names=labelNames)))
f.close()
# afficher le graphe de loss et accuracy
N = numpy.arange(0, 30)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H["loss"], label="train_loss")
plt.plot(N, H["val_loss"], label="val_loss")
plt.plot(N, H["accuracy"], label="train_accuracy")
plt.plot(N, H["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/accuracy")
plt.legend()
plt.savefig("Loss_accuracy_mnist_learn_rate")




