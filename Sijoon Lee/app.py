from flask import Flask, jsonify, render_template #, request
from flask_restful import Api, Resource
import util.predict as pred

app = Flask(__name__)
api = Api(app)
p = pred.Predict()

@app.route('/', methods=['GET'])
def index():
    prediction, year, month, day = p.get()
    prediction = ['{:2.1f}'.format(p) for p in prediction]
    return render_template('home.html', results = prediction, len = len(prediction), year = year, month = month, day = day)



# class Forecast(Resource):
#     def get(self):
#         #Step 1 get the posted data
#         # postedData = request.get_json()

#         #Step 2 read the data
#         #year = postedData["year"]
#         #month = postedData["month"]
#         #day = postedData["day"]

#         #Step 3 Get weather data from gov
#         p = pred.Predict()
#         prediction = p.get()


#         # retJson = {
#         #     "status":200,
#         #     "msg":prediction
#         # }
#         # return jsonify(retJson)

#         return render_template('home.html', results = prediction)

# api.add_resource(Forecast, '/forecast')


if __name__=="__main__":
    app.run(host='0.0.0.0')
