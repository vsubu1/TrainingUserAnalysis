#--------------------------------------------------------------------------------------
# #File name: IRIS_StudentClassification_Views_Prediction_MVRA_1_CLV.py
# Description:  this program performs prediction using Multivariate Regression Analysis
# for view time seconds for course id and level combination.
#--------------------------------------------------------------------------------------

import numpy as np
import statsmodels.api as sm
import numpy as np
import pandas as pd
import os

from src import SQLLite_LoadData as db

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
np.random.seed(2)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

def loadData() :

    # Load Data
    sql = "SELECT user_handle,course_id,author_handle,level,sum(view_time_seconds) view_time_seconds from user_course_views group by user_handle,course_id,author_handle,level"
    data = db.loadDatabySQL(sql)
    data.head()

    # generate sequence numbers for string column values
    data["course_id"] = le.fit_transform(data["course_id"])
    data["level"] = le.fit_transform(data["level"])

    # Explore Data
    #df = pd.DataFrame(data)

    print ("head : ")
    print ("------")
    print (data[1:5])
    print("")
    print ("shape ")
    print ("------")
    print (data.shape)
    print("")
    print ("Describe Result" )
    print ("---------------")
    print (data.describe())
    print("")
    # Prepare X and Y
    X = data[[ "course_id","level"]]
    y = data["view_time_seconds"]


    #Perform linear regression  using formula api
    df2=pd.DataFrame(X,columns=['course_id','level'])
    df2['view_time_seconds']=pd.Series(y)
    return(df2)

def prepareModel(df) :

    image_filename = "./../img/IRIS_StudentClassification_Views_Prediction_MVRA_1_CLV.png"

    # Prepare the regression model and fit X,Y
    model = smf.ols(formula='view_time_seconds~course_id+level', data=df)
    results_formula = model.fit()
    results = results_formula.params

    # Print the model results
    print("Result Parameters")
    print("-----------------")
    print("Intercept" , results.Intercept)
    print("course_id" , results.course_id)
    print("level" , results.level)


    # plot regression model  in 3D for the Training Set
    x_surf, y_surf = np.meshgrid(np.linspace(df.course_id.min(), df.course_id.max(), 100),np.linspace(df.level.min(), df.level.max(), 100))
    onlyX = pd.DataFrame({'course_id': x_surf.ravel(), 'level': y_surf.ravel()})
    fittedY=results_formula.predict(exog=onlyX)

    # Plot the 3D graph for the Training Set
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['course_id'],df['level'],df['view_time_seconds'],c='blue', marker='o', alpha=0.5)
    ax.plot_surface(x_surf,y_surf,fittedY.values.reshape(x_surf.shape), color='None', alpha=0.01)
    ax.set_xlabel('course_id')
    ax.set_ylabel('level')
    ax.set_zlabel('view_time_seconds')
    plt.title('Exploration - View Time Seconds for Course and Level')
    #plt.show()

    # save the plot to directory
    # if image file exists, delete it ##
    if os.path.isfile(image_filename):
        os.remove(image_filename)

    fig.savefig(image_filename)  # save the figure to file
    return(image_filename)


def classify_CLV() :
    print("Running - IRIS_StudentClassification_Views_Prediction_MVRA_1_CLV.py")
    df = loadData()
    image_filename =  prepareModel(df)
    return  (image_filename)
