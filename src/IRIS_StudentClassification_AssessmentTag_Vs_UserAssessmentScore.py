#--------------------------------------------------------------------------------------
# #Filename: IRIS_StudentClassification_AssessmentTag_Vs_UserAssessmentScore.py
# Description: This python code performs K-Means cluster classification for Assesment Tag and
# User Assessment Score with the cluster size of 3
#--------------------------------------------------------------------------------------

import numpy as np

from matplotlib import pyplot as plt
from sklearn import cluster, datasets
from sklearn import preprocessing
import os
from src import SQLLite_LoadData as db

le = preprocessing.LabelEncoder()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Prepare data
def loadData() :

    # Read Data
    df = db.loadData("user_assessment_scores")
    df = df.drop(['user_assessment_date'], 1)

    # Remove header row
    df = df.iloc[1:]

    df["assessment_tag"] = le.fit_transform(df["assessment_tag"])
    df.head()
    df = df.drop(['user_handle'], 1)
    return(df)

def generateReport(df):
    y = df["assessment_tag"].values
    x = df["user_assessment_score"].values
    X = np.array(list(zip(x,y)))

    k_means = cluster.KMeans(n_clusters=3)
    model = k_means.fit(X)
    centroids = k_means.cluster_centers_
    C = k_means.cluster_centers_
    labels = k_means.labels_

    # And we'll visualize it:
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.title('Assessment Score Classification by Assessment Tags')
    plt.xlabel("user_assessment_score")
    plt.ylabel("assessment_tag")
    plt.scatter(x, y, c=model.labels_.astype(float))
    #plt.show()

    # Now, save the plot to directory
    image_filename = "./../img/IRIS_StudentClassification_AssessmentTag_Vs_UserAssessmentScore.png"

    ## if image file exists, delete it ##
    if os.path.isfile(image_filename):
        os.remove(image_filename)

    fig.savefig(image_filename)  # save the figure to file
    plt.close()
    return (image_filename)


def classify_ATUAS() :
    print("Running - IRIS_StudentClassification_AssessmentTag_Vs_UserAssessmentScore.py")
    df = loadData()
    image_filename = generateReport(df)
    return (image_filename)

