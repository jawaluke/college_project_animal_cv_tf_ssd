import cv2
#import imutils
import numpy as np
from subprocess import call
import time
import os
import glob
import smtplib
import base64
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
#from email.MIMEText import MIMEText
from email.mime.text import MIMEText
import sys
import socket
from playsound import playsound
import serial

## ------------------------- tensorflow ------------------------------
#%%bash
#cd models/research/
#!"C:\Users\jawah\OneDrive\Desktop\tensorflow\protoc-3.15.3-win64\bin\protoc.exe" object_detection/protos/*.proto --python_out=.
#
# %%bash
# cd models/research
# pip install .


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name,
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

import pathlib

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
## TEST_IMAGE_PATHS

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy()
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

  return output_dict



def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = image_path #np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  try:
    image_np,classed = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)
    return image_np, classed

  except:
        image_np = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)
        return image_np



  #display(Image.fromarray(image_np))
  #return image_np, classed



## changes 1

#port=serial.Serial("COM3",baudrate=9600, timeout=.3)

# Loading image
cap = cv2.VideoCapture(0)
def mail(name):
    img_data = open("1.jpg", 'rb').read()
    msg = MIMEMultipart()
    From = 'jawaharlakshmanan6@gmail.com'
    To = ["jaiaravindh111399@gmail.com","thomassjerin3@gmail.com"]
    msg['Subject'] = 'subject : Alert!! '+name+' Animal found...'
    msg['From'] = 'jawaharlakshmanan6@gmail.com'
    msg['To'] = "jaiaravindh111399@gmail.com"

    text = MIMEText("test")
    msg.attach(text)
    image = MIMEImage(img_data, name=os.path.basename("1.jpg"))
    msg.attach(image)

    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("jawaharlakshmanan6@gmail.com", "BorntowiN")
    s.sendmail(From, To, msg.as_string())
    s.quit()
    print ('successfully sent the mail')


## <--------------------------------- opencv ---------------------->
## FOR VIDEO CAMERA

import cv2

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 18000)
vid.set(cv2.CAP_PROP_FPS, 25)

while True:

    ret,frame =cap.read()
    try :
        frame, cls = show_inference(detection_model, frame)
        ## changes 2
        ## changes 3
        # DISPLAYING THE BOUND OF PROPABALILITY
        elephant_count = 0
        for i in cls:
            if (i.split()[0][:-1] in ["elephant","zebra"]):#'cat'or'dog'or'horse'or'sheep'or'cow'or'elephant'or'bear'or'zebra'or'giraffe'):
                if float(i.split()[1][:-1]) >= 75.0:
                    print("we found elephant please call the forest ranger")
                    print("elephant count : ", cls)
                    cv2.imwrite("1.jpg", frame)
                    cv2.imshow("resulted picture ",frame)
        #            playsound('aud2.mpeg')
                    mail(i.split()[0][:-1])
         #           port.write(str.encode('A'))

                else:
                    print("something found like an elephant")
          #          port.write(str.encode('C'))
    except:
        frame = show_inference(detection_model, frame)

    # Display the resulting frame
    cv2.imshow("frame", frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
