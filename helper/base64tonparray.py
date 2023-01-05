import base64
import numpy as np
import cv2


def conversion_base64_to_array(base64_str):

    byte_string = base64.b64decode(base64_str)
    img_1d_array = np.fromstring(byte_string, dtype=np.uint8)
    img_array = cv2.imdecode(img_1d_array, cv2.IMREAD_COLOR)

    return img_array

