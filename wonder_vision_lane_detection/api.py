from flask import Flask, request, render_template
import cv2
import io
import base64
import numpy as np
from lane_detection import Detection
from io import BytesIO
from PIL import Image


application = Flask(__name__, static_url_path='/static')
ob = Detection()


def convert_str_to_image(image_64_encode):
    image_io = io.BytesIO()
    image_io.write(base64.b64decode(image_64_encode))
    image = Image.open(image_io)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


@application.route('/')
def get():
    return render_template('home.html')


@application.route('/key_image_search', methods=['POST'])
def upload():
    image_64_encode = (request.form['data'])
    image_64_encode = image_64_encode[image_64_encode.find(',') + 1:]
    image = convert_str_to_image(image_64_encode)
    img = np.array(image)
    scanned = ob.detect(img)
    pil_img = Image.fromarray(scanned)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    return render_template('home.html', org_pic=image_64_encode, pred=new_image_string)


if __name__ == "__main__":
    application.run(host='localhost', port=3000, debug=True)