#--------------------------------------------------------------------------------------
# #Filename: IRIS_StudentClassification_UserInterests.py
# Description: This python code performs K-Means cluster classification for Assesment Tag and
# User Assessment Score with the cluster size of 3
#--------------------------------------------------------------------------------------

from flask import Flask
from flask_restful import Resource, Api
from flask import send_file

from src import IRIS_StudentClassification_Views__AC_5 as ac
from src import IRIS_StudentClassification_Views_CL_2 as cl
from src import  IRIS_StudentClassification_Views_UCL_4 as ucl
from src import IRIS_StudentClassification_Views_UL_1 as ul
from src import  IRIS_StudentClassification_Views__AL_3 as al
from src import IRIS_StudentClassification_User_Vs_AssessmentScore as uas
from src import IRIS_StudentClassification_AssessmentTag_Vs_UserAssessmentScore as atuas
from src import IRIS_StudentClassification_UserInterests as ui
from src import SQLLite_LoadData as db
from src import IRIS_StudentClassification_KNN_1_CL as knn
from src import IRIS_StudentClassification_Views_Prediction_MVRA_1_CLV as clv1
from src import IRIS_StudentClassification_Views_Prediction_MVRA_2_CLV as clv2

from src import HtmlToText as htmltotext
from matplotlib import pyplot as plt
from sklearn import preprocessing

app = Flask(__name__)
api = Api(app)

le = preprocessing.LabelEncoder()

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

@app.route('/')
class Views_User_Level(Resource):
    def get(self):
        filename = ul.classify_UL()
        return send_file(filename, mimetype='image/gif')

class Views_Course_Level(Resource):
    def get(self):
        filename = cl.classify_CL()
        return send_file(filename, mimetype='image/gif')

class Views_Author_Level(Resource):
    def get(self):
        filename =   al.classify_AL()
        return send_file(filename, mimetype='image/gif')

class Views_User_Course_Level(Resource):
    def get(self):
        filename = ucl.classify_UCL()
        return send_file(filename, mimetype='image/gif')

class Views_Author_Course(Resource):
    def get(self):
        filename = ac.classify_AC()
        return send_file(filename, mimetype='image/gif')

class Score_AssessmentTag_UserAssessmentScore(Resource):
    def get(self):
        filename = atuas.classify_ATUAS()
        return send_file(filename, mimetype='image/gif')

class Score_User_AssessmentScore(Resource):
    def get(self):
        filename = uas.classify_UAS()
        return send_file(filename, mimetype='image/gif')

class Score_User_Interests(Resource):
    def get(self):
        filename = ui.classify_UI()
        return send_file(filename, mimetype='image/gif')

class KNN(Resource):
    def get(self):
        results = knn.predict_CL()
        results = htmltotext.html2text(results)
        return (results)

class Time_Course_Level_ViewTime(Resource):
    def get(self):
        filename = clv1.classify_CLV()
        return send_file(filename, mimetype='image/gif')

class Time_Course_Level_PredictTime(Resource):
    def get(self):
        filename = clv2.predict_CLV()
        return send_file(filename, mimetype='image/gif')

# Classification algorithm implementation

api.add_resource(Views_User_Level, '/classify/ul')  # Route_1
api.add_resource(Views_Course_Level, '/classify/cl')  # Route_2
api.add_resource(Views_Author_Level, '/classify/al')  # Route_3
api.add_resource(Views_User_Course_Level, '/classify/ucl')  # Route_4
api.add_resource(Views_Author_Course, '/classify/ac')  # Route_5
api.add_resource(Score_AssessmentTag_UserAssessmentScore, '/classify/atuas')  # Route_6
api.add_resource(Score_User_AssessmentScore, '/classify/uas')  # Route_7
api.add_resource(Score_User_Interests, '/classify/ui')  # Route_8

# Prediction algorithm implementation
api.add_resource(KNN, '/predict/cl')  # Route_9
api.add_resource(Time_Course_Level_ViewTime, '/classify/clv')  # Route_10
api.add_resource(Time_Course_Level_PredictTime, '/predict/clv')  # Route_11

if __name__ == '__main__':
    app.run(port='5002')