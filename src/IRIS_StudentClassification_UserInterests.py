#--------------------------------------------------------------------------------------
# #Filename: IRIS_StudentClassification_UserInterests.py
# Description: This python code performs K-Means cluster classification for Interest Tag and
# User Handle with the cluster size of 3
#--------------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster, datasets
from src import SQLLite_LoadData as db
from sklearn import preprocessing
import os

le = preprocessing.LabelEncoder()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Read Data

def loadData() :

    df = db.loadData("user_interests")
    df.head()

    # Remove header row
    df = df.iloc[1:]

    # drop unwanted columns
    df = df.drop(['date_followed'], 1)

    # convert interest_tag to serial numbers
    df["interest_tag"] = le.fit_transform(df["interest_tag"])
    return(df)

def prepareModel(df) :
    y = df["user_handle"].values
    x = df["interest_tag"].values
    X = np.array(list(zip(x,y)))

    k_means = cluster.KMeans(n_clusters=6)
    model = k_means.fit(X)

    centroids = k_means.cluster_centers_
    print("Centroids")
    print(centroids)

    labels = k_means.labels_
    print(labels)

    return(model,x,y)

def generateReport(model,x,y) :
    # And we'll visualize it:
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.title('User Interests by Interest Tag')
    plt.xlabel("interest_tag")
    plt.ylabel("user_handle")
    plt.scatter(x, y, c=model.labels_.astype(float))
    #plt.show()

    # Now, save the plot to directory
    image_filename = "./../img/IRIS_StudentClassification_Interests.png"

    ## if image file exists, delete it ##
    if os.path.isfile(image_filename):
        os.remove(image_filename)

    fig.savefig(image_filename)  # save the figure to file
    plt.close()
    return (image_filename)

def classify_UI() :
    print("Running - IRIS_StudentClassification_UserInterests.py")
    df = loadData()
    (model,x,y) = prepareModel(df)
    image_filename = generateReport(model,x,y)
    return (image_filename)
