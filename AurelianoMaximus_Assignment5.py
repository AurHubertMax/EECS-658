
'''
  Name of program: ImbalancedDatasets
  Author: Aureliano Hubert Maximus
  Creation Date: Oct 26 2023
  Description:
    this program uses the dataset in file "imbalanced iris.csv" and first prints out the Accuracy Score, Class Balanced Accuracy, Balanced Accuracy, and Confusion Matrix of the dataset. Then,
    it also balances the imbalanced dataset with random oversampling using SMOTE, and ADASYn oversampling and prints out their respective Accuracy Score, and Confusion Matrix. Then,
    it balances the imbalanced dataset with random undersampling using Cluster, and Tomek Links and prints out their respective Accuracy Score, and Confusion Matrix.
  Collaborators:
  Prof. David Johnson (EECS 658 Slides and email)
  ChatGPT
  '''

  # load libraries
  from pandas import read_csv
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import accuracy_score
  from sklearn.metrics import recall_score
  from sklearn.metrics import precision_score
  from sklearn.datasets import load_iris
  from sklearn import datasets
  from sklearn import preprocessing
  from sklearn.neural_network import MLPClassifier
  import numpy as np
  from imblearn.over_sampling import RandomOverSampler
  from imblearn.over_sampling import SMOTE
  from imblearn.over_sampling import ADASYN
  from imblearn.under_sampling import RandomUnderSampler
  from imblearn.under_sampling import ClusterCentroids
  from imblearn.under_sampling import TomekLinks


  #load dataset
  url = "imbalanced iris.csv"
  names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
  dataset = read_csv(url, names=names)

  #Create Arrays for Features and Classes
  array = dataset.values
  X = array[:,0:4] #contains flower features (petal length, etc..)
  y = array[:,4] #contains flower names

  #Split Data into 2 Folds for Training and Test
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.5, random_state=1)

  #handles all the models and their data
  def regModel(model, name):

    model.fit(X_Fold1, y_Fold1) #training for fold 1
    pred1 = model.predict(X_Fold2) #testing for fold 1

    model.fit(X_Fold2, y_Fold2) #training for fold 2
    pred2 = model.predict(X_Fold1) #testing for fold2

    actual = np.concatenate([y_Fold2, y_Fold1]) #combine actuals
    predicted = np.concatenate([pred1, pred2]) #combine predicted

    recall = recall_score(actual, predicted, average=None) #calculates recall
    precision = precision_score(actual, predicted, average=None) #calculates precision

    falsePos = confusion_matrix(actual, predicted).sum(axis=0) - np.diag(confusion_matrix(actual, predicted)) #calculates false positive data, used to find the average
    falseNeg = confusion_matrix(actual, predicted).sum(axis=1) - np.diag(confusion_matrix(actual, predicted)) #calculates false negative data, used to find the average
    truePos = np.diag(confusion_matrix(actual, predicted)) #calculates true positive data, used to find the average
    trueNeg = confusion_matrix(actual, predicted).sum() - (falsePos + falseNeg + truePos) #calculates true negative data, used to find the average
    spec = trueNeg / (trueNeg + falsePos) #calculates the specificity, used to find the average

    average1 = [None] * len(recall)
    minRecPrec = [None] * len(recall)

    for i in range(len(recall)):
      minRecPrec[i] = min(recall[i], precision[i])
      average1[i] = (recall[i] + spec[i]) / 2

    average1 = sum(average1) / len(average1)
    average2 = sum(minRecPrec) / len(minRecPrec)

    #prints the results
    print('%s' % name)
    accuracy = accuracy_score(actual, predicted) #calculate accuracy score
    print('Accuracy Score: ' + str(round(accuracy, 4))) #prints accuracy score

    #only Neural Network model would need to print these
    if (name == 'Neural Network'):
      print('Class Balanced Accuracy: ' + str(round(average1, 3)))
      print('Balanced Accuracy: ' + str(round(average2, 3)))

    print('Confusion Matrix: ')
    print(confusion_matrix(actual, predicted)) #prints confusion matrix
    print("")

  #Neural Network
  regModel(MLPClassifier(max_iter=1000), 'Neural Network')

  #Random Oversampler
  #balances dataset with Random Oversampling
  X_new, y_new = RandomOverSampler(random_state=0).fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'Random Oversampling')

  #SMOTE
  X_new, y_new = SMOTE().fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'SMOTE')

  #ADASYN
  X_new, y_new = ADASYN(random_state=0, sampling_strategy='minority').fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'ADASYN')

  #Random Undersampler
  #balances dataset with Random Undersampling
  X_new, y_new = RandomUnderSampler(random_state=0).fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'Random Undersampling')

  #Cluster Centroids
  X_new, y_new = ClusterCentroids(random_state=0).fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'Cluster Centroid')

  #Tomek Links
  X_new, y_new = TomekLinks().fit_resample(X, y)
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X_new, y_new, test_size=0.5, random_state=1)
  regModel(MLPClassifier(max_iter=1000), 'Tomek Links')