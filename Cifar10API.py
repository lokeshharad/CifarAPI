

from flask import Flask, jsonify, request
from flask_cors import CORS

from keras.models import load_model

from helper import config as cf
from helper import base64tonparray
from helper import log_file_creation as log

import numpy as np
import json
import cv2
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
cors = CORS(app)


@app.route('/', methods=["GET"])
def home():
    return "<h1>connection is successful<h1>"
# ******************************************************************************


@app.route('/predict', methods=["POST"])
def CifarAPI():

    doc_data = dict()
    doc_data["ResponseCode"] = ""
    doc_data["ResponseMessage"] = ""

    response_data = dict()
    response_data["Document"] = ""
    response_data["Probability"] = "0"

    doc_data["Response"] = response_data
    print(doc_data)

    try:
        print(type(request.data))
        # print(request.data)
        request_param = json.loads(request.data)

        log.logger.info(request.data)

        if "image" not in request_param.keys():

            message = "Missing Key \"image\" Parameter."
            doc_data["ResponseCode"] = "999"
            doc_data["Response"] = "BAD REQUEST"
            doc_data["ResponseMessage"] = message

            log.logger.error(message)

        elif (not isinstance(request_param["image"], str)) and (len(request_param["image"].strip()) == 0):

            message = "Inappropriate Data in \"image\" Parameter."
            doc_data["ResponseCode"] = "999"
            doc_data["Response"] = "BAD REQUEST"
            doc_data["ResponseMessage"] = message

            log.logger.error(message)

        else:
            image = base64tonparray.conversion_base64_to_array(request_param["image"])
            print(image.shape)
            # t_img = cv2.imread("pan_lokesh.png")
            # print(t_img.shape)

            t_img_resize = np.array(cv2.resize(image, (32, 32)))
            print("resize image: ", t_img_resize.shape)

            t_pred_arr = model.predict(t_img_resize.reshape((1, 32, 32, 3)))
            # t_pred_arr = model.predict(t_img_resize)
            print(t_pred_arr)
            t_pred_prob = t_pred_arr.max(axis=1)[0]
            t_pred = t_pred_arr.argmax(axis=1)[0]

            print(t_pred)
            print(cf.class_label[t_pred])
            print

            response_data["Probability"] = str(t_pred_prob)
            response_data["Document"] = cf.class_label[t_pred]
            doc_data["Response"] = response_data
            doc_data["ResponseMessage"] = "SUCCESS"
            doc_data["ResponseCode"] = "000"

            log.logger.info(json.dumps(doc_data))

    except Exception as ex:

        message = ex
        doc_data["ResponseMessage"] = message
        doc_data["ResponseCode"] = "999"
        doc_data["Response"] = {}

        log.logger.error(message)

    return jsonify(doc_data)


if __name__ == '__main__':

    model_path = cf.model_path
    model = load_model(model_path)
    log.logger.info("Model is loaded.")
    print(model.summary())

    app.run(host='0.0.0.0', port=cf.port_value, debug=True)
