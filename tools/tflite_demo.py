# Requirements listed in requirements/tflite.txt

import cv2
import tensorflow as tf
import numpy as np


def transform(image: np.ndarray, img_w=128, img_h=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_AREA)
    image = ((image.astype(np.float32) / 255.0) - 0.5) / 0.5
    return np.transpose(image, (2, 0, 1))


if __name__ == '__main__':
    # Load the TFLite model in TFLite Interpreter
    interpreter = tf.lite.Interpreter('tflite/parseq.tflite')

    # There is only 1 signature defined in the model,  so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    tf_model = interpreter.get_signature_runner()

    img_path = "demo_images/ic15_word_26.png"
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_processed = transform(image)

    image_tensor = tf.convert_to_tensor(image_processed)
    image_tensor = tf.expand_dims(image_tensor, 0)

    input_names = [key for key in tf_model._inputs.keys()]
    inputs = { input_names[0]: image_tensor }

    output = tf_model(**inputs)
    print(output)
