'''
  Name of program: NBClassifier.py
  Author: Aureliano Hubert Maximus
  Description:
    this program uses the dataset in file "iris.csv", to predict the different
    types of flowers (Iris-setosa, Iris-versicolor, and Iris-virginica),
    and separates the data into 2 Folds, where Fold 1 will be used to train the
    ML model, while Fold 2 is used for testing. After that, Fold 2 is used for
    training while Fold 1 is used for testing. The final results of the program
    is then printed in the console.
  
  Collaborators:
    Prof. David Johnson (EECS 658 Slides)
  '''


  # load libraries
  from pandas import read_csv
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import accuracy_score
  from sklearn.naive_bayes import GaussianNB
  import numpy as np

  #load dataset
  url = "iris.csv"
  names = ['sepal-length', 'sepal-width', 'petal-length',
  'petal-width', 'class']
  dataset = read_csv(url, names=names)


  #Create Arrays for Features and Classes
  array = dataset.values
  X = array[:,0:4] #contains flower features (petal length, etc..)
  y = array[:,4] #contains flower names


  #Split Data into 2 Folds for Training and Test
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y,
  test_size=0.50, random_state=1)

  model = GaussianNB() #select ML model
  model.fit(X_Fold1, y_Fold1) #first fold training
  pred1 = model.predict(X_Fold2) #first fold testing
  model.fit(X_Fold2, y_Fold2) #second fold training
  pred2 = model.predict(X_Fold1) #second fold testing
  actual = np.concatenate([y_Fold2, y_Fold1]) #actual classes
  predicted = np.concatenate([pred1, pred2]) #predicted classes
  print(accuracy_score(actual, predicted)) #accuracy
  print(confusion_matrix(actual, predicted)) #confusion matrix
  print(classification_report(actual, predicted)) #P, R, & F1

  print("Hello World!") #prints out "Hello World"