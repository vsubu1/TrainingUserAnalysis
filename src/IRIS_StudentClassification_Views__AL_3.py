#--------------------------------------------------------------------------------------
# #Filename: IRIS_StudentClassification_Views__AL_3.py
#Description: This python code K-Means cluster classification for author and level
#with cluster size as 3
#--------------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
import os

from src import SQLLite_LoadData as db

le = preprocessing.LabelEncoder()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Read Data
def loadData() :
    df = db.loadData("user_course_views")
    df.head()

    # select only first 5 columns
    df = df.iloc[:,0:5]

    # drop unwanted columns for this analysis
    df = df.drop(['view_date','user_handle'], 1)

    # Convert the string data to numeric series
    df["course_id"] = le.fit_transform(df["course_id"])
    df["level"] = le.fit_transform(df["level"])

    # Remove header row
    df = df.iloc[1:]
    return(df)

def prepareModel(df) :

    x = df['author_handle']
    y = df['level']
    X = np.array(list(zip(x,y)))

    k_means = cluster.KMeans(n_clusters=3)
    model = k_means.fit(X)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_
    return(model,x,y)

def generatePlot(model,x,y) :
    # And we'll visualize it:
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.title('Classification of User Views by Author hanldes and Levels')
    plt.xlabel("author_handle")
    plt.ylabel("level")
    plt.scatter(x, y, c=model.labels_.astype(float))
    #plt.show()

    # Now, save the plot to directory
    image_filename = "./../img/IRIS_StudentClassification_Views__AL_3.png"

    ## if image file exists, delete it ##
    if os.path.isfile(image_filename):
        os.remove(image_filename)

    fig.savefig(image_filename)  # save the figure to file
    plt.close()
    return (image_filename)


def classify_AL() :

    print("Running  - IRIS_StudentClassification_Views__AL_3.py")

    df = loadData()
    model,x,y = prepareModel(df)
    image_filename = generatePlot(model,x,y)
    return(image_filename)
