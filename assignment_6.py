'''
  Name of program: Assignment 6
  Author: Aureliano Hubert Maximus
  Creation Date: Nov 9 2023
  Description:
    this program uses the dataset in file "iris.csv" and applies K-Means Clustering,
    and GMM to print out a plot, its accuracy score, and confusion matrix.
  Collaborators:
  Prof. David Johnson (EECS 658 Slides)
  ChatGPT
  '''
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture

url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)



#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

#Changes class names to number representation
labelEncoder = LabelEncoder()
y_true = labelEncoder.fit_transform(y)

#Plot function given by Prof. Johnson
def plot_graph(arr, name):
  plt.plot(range(1, 21), arr, marker='o')
  plt.title(name + ' vs. k')
  plt.xlabel('k')
  plt.xticks(np.arange(1, 21, 1))
  plt.ylabel(name)
  plt.show()

#Function to print accuracy score, and confusion matrix
def print_Acc_Con(name, actual, predicted):
  print('%s' % name)
  accuracy = accuracy_score(actual, predicted) #calculate accuracy score
  print('Accuracy Score: ' + str(round(accuracy, 4))) #prints accuracy score
  print('Confusion Matrix: ')
  print(confusion_matrix(actual, predicted)) #prints confusion matrix
  print("")


#Function for K means clustering
def part1(XData, elbow_k):
    rec_err = []

    # Iterate through k values for K means
    for k in range(1, 21):
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        model.fit(XData)
        clusters = model.predict(XData)

        rec_err.append(mean_squared_error(XData, model.cluster_centers_[clusters]))

    #plot the graph
    plot_graph(rec_err, 'Reconstruction Error')

    #Fit K means with elbow
    model = KMeans(n_clusters=elbow_k, random_state=0, n_init="auto")
    model.fit(XData)
    clusters = model.predict(XData)

    #Sets cluster labels based on majority class
    cluster_labels = {}
    for cluster in range(elbow_k):
        majority = np.argmax(np.bincount(y_true[clusters == cluster]))
        cluster_labels[cluster] = majority

    #Maps the cluster_labels to true labels
    pred = np.array([cluster_labels[cluster] for cluster in clusters])

    #Calls function to print results
    print_Acc_Con("K-Means Clustering", y_true, pred)

def part2(XData):
  # AIC PART
  aic_score = []

  #Iterate through k values for AIC GMM
  for k in range(1, 21):
    model = GaussianMixture(n_components=k, covariance_type='diag', random_state=0)
    model.fit(XData)
    aic_score.append(model.aic(XData))

  #Plots the AIC graph
  plot_graph(aic_score, 'AIC')

  #Finds elbow_k for aic
  aic_elbow_k = findGmmElbow(aic_score)

  #Fits AIC GMM with aic_elbow_k
  model = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag', random_state=0)
  model.fit(XData)
  clusters = model.predict(XData)

  #Sets cluster labels based on majority class
  cluster_labels = {}
  for cluster in range(aic_elbow_k):
    majority = np.argmax(np.bincount(y_true[clusters == clusters]))
    cluster_labels[cluster] = majority

  #Maps the cluster_labels to true labels
  pred_AIC = np.array([cluster_labels[cluster] for cluster in clusters])

  #Calls function to print results
  print_Acc_Con('AIC', y_true, pred_AIC)


 # BIC PART
  bic_score = []

  #Iterate through k values for BIC GMM
  for k in range(1, 21):
    model = GaussianMixture(n_components=k, covariance_type='diag', random_state=0)
    model.fit(XData)
    bic_score.append(model.bic(XData))

  #Plots the AIC graph
  plot_graph(bic_score, 'BIC')

  #Finds elbow_k for bic
  bic_elbow_k = findGmmElbow(bic_score)

  #Fits BIC GMM with bic_elbow_k
  model = GaussianMixture(n_components=bic_elbow_k, covariance_type='diag', random_state=0)
  model.fit(XData)
  clusters = model.predict(XData)

  #Sets cluster labels based on majority class
  cluster_labels = {}
  for cluster in range(bic_elbow_k):
    majority = np.argmax(np.bincount(y_true[clusters == clusters]))
    cluster_labels[cluster] = majority

  #Maps the cluster_labels to true labels
  pred_BIC = np.array([cluster_labels[cluster] for cluster in clusters])

  #Calls function to print results
  print_Acc_Con('BIC', y_true, pred_BIC)

#Function to give elbow of a curve
def findGmmElbow(arr):
  #Calculate difference between consecutive elements in array
  diff = np.diff(arr)

  #Find index of maximum difference & add 2, since index starts from 0
  return np.where(diff == max(diff))[0][0] + 2



elbow_k = 3
part1(X, elbow_k)

part2(X)