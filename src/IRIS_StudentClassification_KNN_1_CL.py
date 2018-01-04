#--------------------------------------------------------------------------------------
# Filename: IRIS_StudentClassification_KNN_1_CL.py
# Description:
# This python code is developed for predicting multivariate classification model data.
# It reads Iris data set and splits them into train set and test set
# Using KNN Neighbours classification, it trains the model using the test set data along with test set
# classified results.
# Then it predicts the classification of the test set.
#--------------------------------------------------------------------------------------

from sklearn import preprocessing

from src import SQLLite_LoadData as db

le = preprocessing.LabelEncoder()

import numpy as np
import pandas as pd


def predict_CL() :

    print("Running  - IRIS_StudentClassification_KNN_1_IRIS_StudentClassification_KNN_1_CLCL.py")
    # Load Data
    sql = "SELECT user_handle,course_id,author_handle,level,sum(view_time_seconds) view_time_seconds from user_course_views group by user_handle,course_id,author_handle,level"
    df = db.loadDatabySQL(sql)
    df.head()

    # Remove header row
    data = df.iloc[1:]

    # Remove last row as it contains the column header in string format
    data = data.drop(data.index[len(data)-1])

    data["course_id"] = le.fit_transform(data["course_id"])
    data["level"] = le.fit_transform(data["level"])
    data = data.apply(pd.to_numeric)

    # Prepare X and Y
    iris_X = data[[ 'author_handle','course_id']]
    iris_y = data['level']

    # consider 400 rows for taining set and 100 rows of data for test set
    iris_X_train = iris_X[:-400]
    iris_y_train = iris_y[:-400]
    iris_X_test  = iris_X[-100:]
    iris_y_test  = iris_y[-100:]

    # prepare the KNN classifier model
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(iris_X_train, iris_y_train)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')

    # predict using test set
    pr = knn.predict(iris_X_test)
    #print(pr)

    # display the results
    dfFinal = pd.DataFrame(iris_X_test)
    dfFinal['Actual result'] = iris_y_test
    dfFinal['Predicted Result'] = pr

    # Return first 10 rows for actual and predicted set of rows
    df1=dfFinal.head()
    htmltext = df1.to_html()
    print(df1)
    return(htmltext)

