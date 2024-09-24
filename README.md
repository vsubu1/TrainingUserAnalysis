# TrainingUserAnalysis

This project performs classification and prediction using machine learning algorithms for training users dataset. This project uses REST API using Flask and SQLite DB which is preloaded with the set of user training data.
To Setup Project

    Download the repository
    Unzip training.db.zip file and place the training.db file under db folder.
    Open PyCharm, create a new project and copy the files fromdownloaded repository into the new project created.
    Run RestAPI.py file for executing the machine learning algorithms.
    Open the URL http://127.0.0.1:5002/ in a new browser window
    Start the execution with the url http://127.0.0.1:5002/classify/ul to view the classification report.
    Run the remaining requests as given in the Rest API calls - Execution instructions section.

Machine Learning algorithms used

    K-Means Clustering
    K-Nearest Neighbour algorithm
    Multivariate regression analysis

Rest API calls - Execution instructions

A. Classification algorithm implementation

    http://127.0.0.1:5002/classify/ul - generates classification of user views based on course levels
    http://127.0.0.1:5002/classify/cl - generates classification of courses based on course levels
    http://127.0.0.1:5002/classify/al - generates classification of authors based on course levels
    http://127.0.0.1:5002/classify/ucl - generates Classification of user views by courses and levels
    http://127.0.0.1:5002/classify/ac - generates classification of authors based on course levels
    http://127.0.0.1:5002/classify/atuas - generates classification of assessment tags based on user assessment score
    http://127.0.0.1:5002/classify/uas - generates classification of users based on assessment scores
    http://127.0.0.1:5002/classify/ui - generates classification of users based on interest tags
    http://127.0.0.1:5002/classify/clv - generates data exploration of users based on course, level and view time seconds which is subsequently used for prediction using Multi Variate Regression Analysis.

B. Prediction algorithm implementation

    http://127.0.0.1:5002/predict/cl - predicts courses levels for a set of courses using KNN.
    http://127.0.0.1:5002/predict/clv - predicts course view time using Multivariate Regression Analysis for 3 sample test data.
