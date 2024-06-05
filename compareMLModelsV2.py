'''
  Name of program: CompareMLModelsV2
  Author: Aureliano Hubert Maximus
  Creation Date: Sep 28 2023
  Description:
    this program uses the dataset in file "iris.csv" and applies (0 to 3 degrees)
    Polynomial Regression, Gaussian, KNN, LDA, QDA, SVM, Decition Tree, Random Forest,
    Extra Trees, and NN machine learning models, and prints out their respective
    accuracy score and confusion matrix by separating the "iris.csv" into 2 Folds.
    Some encoding is required to use the Regression models, by transforming the names
    of the flowers into numbers, using the LabelEncoder library in Sklearn.
  Collaborators:
  Prof. David Johnson (EECS 658 Slides and email)
  '''

  # load libraries
  from pandas import read_csv
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import classification_report
  from sklearn.metrics import confusion_matrix
  from sklearn.metrics import accuracy_score
  from sklearn.datasets import load_iris
  from sklearn import datasets
  from sklearn.naive_bayes import GaussianNB
  from sklearn.linear_model import LinearRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  from sklearn.preprocessing import LabelEncoder
  from sklearn import preprocessing
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn import tree
  from sklearn.tree import DecisionTreeClassifier
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.ensemble import ExtraTreesClassifier
  from sklearn.neural_network import MLPClassifier
  from sklearn.svm import LinearSVC
  import numpy as np


  #load dataset
  url = "iris.csv"
  names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
  dataset = read_csv(url, names=names)

  #Create Arrays for Features and Classes
  array = dataset.values
  X = array[:,0:4] #contains flower features (petal length, etc..)
  y = array[:,4] #contains flower names

  #Use encoded training and validation values for prediction on linear regression
  encoder = preprocessing.LabelEncoder()
  encoder.fit(y) #make encoder fit to y's data, which contains the flower names

  #Split Data into 2 Folds for Training and Test
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, encoder.transform(y), test_size=0.50, random_state=1)


  #Algorithm for regression models
  def regModel(name, model):
  #Fit and transform data sets according to the regression degree
    poly_reg = None
    if (name == "Linear Regression"):
      poly_reg = PolynomialFeatures(degree=1)
    elif(name == "2 Degree Polynomial Regression"):
      poly_reg = PolynomialFeatures(degree=2)
    elif(name == "3 Degree Polynomial Regression"):
      poly_reg = PolynomialFeatures(degree=3)

  #Linear Regression
    X_Poly1 = poly_reg.fit_transform(X_Fold1) #makes Fold1 applied to the specific polynomial regression
    X_Poly2 = poly_reg.fit_transform(X_Fold2) #makes Fold2 applied to the specific polynomial regression
    model.fit(X_Poly1, y_Fold1) #first fold training
    pred1 = model.predict(X_Poly2).round() #first fold testing
    pred1 = np.where(pred1 >= 3.0, 2.0, pred1) #first prediction results
    pred1 = np.where(pred1 <= -1.0, 0.0, pred1)

    model.fit(X_Poly2, y_Fold2) #second fold training
    pred2 = model.predict(X_Poly1).round() #second fold testing
    pred2 = np.where(pred2 >= 3.0, 2.0, pred2) #second prediction results
    pred2 = np.where(pred2 <= -1.0, 0.0, pred2)

    actual = np.concatenate([y_Fold2, y_Fold1]) #append y fold 2 and 1 together
    predicted = np.concatenate([pred1, pred2]) #append both prediction results together

    #print the results
    print('%s' % name)
    accuracy = accuracy_score(actual, predicted) #calculate accuracy score
    print('Accuracy Score: ' + str(round(accuracy,3))) #prints accuracy score
    print('Confusion Matrix: ')
    print(confusion_matrix(actual, predicted)) #prints confusion matrix
    print("")

  regModel("Linear Regression", LinearRegression()) #calls regModel to calculate and display "Linear Regression" results
  regModel("2 Degree Polynomial Regression", LinearRegression()) #calls regModel to calculate and display "2 Degree Linear Regression" results
  regModel("3 Degree Polynomial Regression", LinearRegression()) #calls regModel to calculate and display "3 Degree Linear Regression" results


  #Split Data into 2 Folds for Training and Test
  X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y,test_size=0.50, random_state=1)

  #Algorithm for other models
  def MLModel(model):
    model.fit(X_Fold1, y_Fold1) #first fold training
    pred1 = model.predict(X_Fold2) #first fold testing

    model.fit(X_Fold2, y_Fold2) #first fold training
    pred2 = model.predict(X_Fold1) #first fold testing
    y_true = np.concatenate([y_Fold1, y_Fold2]) #append y fold1 and 2 together
    y_pred = np.concatenate([pred2, pred1]) #append both prediction results together

    #print the results
    print('%s' % model) #print the name of the model
    accuracy = accuracy_score(y_true, y_pred) #calculate the accuracy score
    print('Accuracy Score: ' + str(round(accuracy,3))) #prints the accuracy score
    print('Confusion Matrix: ')
    print(confusion_matrix(y_true, y_pred)) #prints the confusion matrix
    print("")

  MLModel(GaussianNB()) #calls regModel to calculate and display "GaussianNB" results
  MLModel(KNeighborsClassifier()) #calls regModel to calculate and display "KNN" results
  MLModel(LinearDiscriminantAnalysis()) #calls regModel to calculate and display "LDA" results
  MLModel(QuadraticDiscriminantAnalysis()) #calls regModel to calculate and display "QDA" results
  MLModel(LinearSVC(max_iter = 10000)) #calls regModel to calculate and display "SVC" results, with a huge max iterations for convergence purposes
  MLModel(tree.DecisionTreeClassifier()) #calls regModel to calculate and display "Decision Tree" results
  MLModel(RandomForestClassifier()) #calls regModel to calculate and display "Random Forest" results
  MLModel(ExtraTreesClassifier()) #calls regModel to calculate and display "Extra Trees" results
  MLModel(MLPClassifier(max_iter = 10000)) #calls regModel to calculate and display "Neural Network" results, with a huge max iterations for convergence purposes

