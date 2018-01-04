#--------------------------------------------------------------------------------------
# Filename: IRIS_StudentClassification_User_Vs_AssessmentScore.py
# Description:
# This python code is developed for classifying Users based on Assessment Score using
#  K-Means Classification.
#--------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import cluster, datasets
from sklearn import preprocessing
import os
from src import SQLLite_LoadData as db

le = preprocessing.LabelEncoder()
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# load and prepare data
def loadData() :

    print("Running  - IRIS_StudentClassification_KNN_1_IRIS_StudentClassification_KNN_1_CLCL.py")

    # Load Data
    df = db.loadData("user_assessment_scores")

    # remove unwanted columns
    df = df.drop(['user_assessment_date'], 1)

    # conver the column values to sequence numbers
    df["assessment_tag"] = le.fit_transform(df["assessment_tag"])
    df = df.drop(['assessment_tag'], 1)

    # Remove header row
    df = df.iloc[1:]
    
    return(df)

# prepare model and generate classification report for user and assessment score parameters
def generateReport(df) :

    x = df["user_handle"].values
    y = df["user_assessment_score"].values
    X = np.array(list(zip(x,y)))

    k_means = cluster.KMeans(n_clusters=3)
    model = k_means.fit(X)

    centroids = k_means.cluster_centers_
    print(centroids)

    labels = k_means.labels_
    print(labels)

    # And we'll visualize it:

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.title('Assessment Score Classification by Users')
    plt.xlabel("user_handle")
    plt.ylabel("user_assessment_score")
    plt.scatter(x, y, c=model.labels_.astype(float))
    #plt.show()

    # Now, save the plot to directory
    image_filename = "./../img/IRIS_StudentClassification_Assessment_Vs_User.png"

    ## if image file exists, delete it ##
    if os.path.isfile(image_filename):
        os.remove(image_filename)

    fig.savefig(image_filename)  # save the figure to file
    plt.close()
    return (image_filename)

def classify_UAS() :
    print("Running - IRIS_StudentClassification_Assessment_Vs_User.py")
    df = loadData()
    image_filename = generateReport(df)
    return (image_filename)

