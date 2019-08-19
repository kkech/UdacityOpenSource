from flask import Flask
from flask import jsonify
import Model as m


app = Flask(__name__)

global model

model = m.init()


@app.route('/')
def welcome():
    return "Welcome you need a secret key to access this api"


@app.route('/<key>/<num>')
def request(key, num):

    k = '01732033963'

    if key != k:
        li = {"output": "Invalid key"}
        return jsonify(li)

    output = m.process(model=model, num=num)
    list = {"output": output}
    return jsonify(list)


if __name__ == '__main__':
    app.run()
