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




## <--------------------  tensorflow mobile ssd   --------------------->

import pathlib


class Tensorflow_mobile_SSD:
    def __init__(self):
        # patch tf1 into `utils.
        utils_ops.tf = tf.compat.v1
        # Patch the location of gfile
        tf.gfile = tf.io.gfile
        # List of the strings that is used to add correct label for each box.
        self.PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        self.PATH_TO_TEST_IMAGES_DIR = pathlib.Path('object_detection/test_images')
        self.TEST_IMAGE_PATHS = sorted(list(self.PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        ## TEST_IMAGE_PATHS

        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.detection_model = self.load_model(model_name)
        print(self.detection_model.signatures['serving_default'].inputs)
        self.detection_model.signatures['serving_default'].output_dtypes
        self.detection_model.signatures['serving_default'].output_shapes



    def load_model(self, model_name):
      self.base_url = 'http://download.tensorflow.org/models/object_detection/'
      self.model_file = model_name + '.tar.gz'
      self.model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=self.base_url + self.model_file,
        untar=True)

      self.model_dir = pathlib.Path(self.model_dir)/"saved_model"

      self.model = tf.saved_model.load(str(self.model_dir))

      return self.model

    def run_inference_for_single_image(self, model, image):
      self.image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      self.input_tensor = tf.convert_to_tensor(self.image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      self.input_tensor = self.input_tensor[tf.newaxis,...]

      # Run inference
      self.model_fn = model.signatures['serving_default']
      self.output_dict = self.model_fn(self.input_tensor)

      # All outputs are batches tensors.
      # Convert to numpy arrays, and take index [0] to remove the batch dimension.
      # We're only interested in the first num_detections.
      self.num_detections = int(self.output_dict.pop('num_detections'))
      self.output_dict = {key:value[0, :self.num_detections].numpy()
                     for key,value in self.output_dict.items()}
      self.output_dict['num_detections'] = self.num_detections

      # detection_classes should be ints.
      self.output_dict['detection_classes'] = self.output_dict['detection_classes'].astype(np.int64)

      # Handle models with masks:
      if 'detection_masks' in self.output_dict:
        # Reframe the the bbox mask to the image size.
        self.detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  self.output_dict['detection_masks'], self.output_dict['detection_boxes'],
                   self.image.shape[0], self.image.shape[1])
        self.detection_masks_reframed = tf.cast(self.detection_masks_reframed > 0.5,
                                           tf.uint8)
        self.output_dict['detection_masks_reframed'] = self.detection_masks_reframed.numpy()

      return self.output_dict

    def show_inference(self, model, image_path):
       # the array based representation of the image will be used later in order to prepare the
       # result image with boxes and labels on it.
       self.image_np = image_path #np.array(Image.open(image_path))
       # Actual detection.
       self.output_dict = self.run_inference_for_single_image(model, self.image_np)
       # Visualization of the results of a detection.
       try:
         self.image_np,self.classed = vis_util.visualize_boxes_and_labels_on_image_array(
               self.image_np,
               self.output_dict['detection_boxes'],
               self.output_dict['detection_classes'],
               self.output_dict['detection_scores'],
               self.category_index,
               instance_masks=self.output_dict.get('detection_masks_reframed', None),
               use_normalized_coordinates=True,
               line_thickness=8)
         return self.image_np, self.classed

       except:
             self.image_np = vis_util.visualize_boxes_and_labels_on_image_array(
               self.image_np,
               self.output_dict['detection_boxes'],
               self.output_dict['detection_classes'],
               self.output_dict['detection_scores'],
               self.category_index,
               instance_masks= self.output_dict.get('detection_masks_reframed', None),
               use_normalized_coordinates=True,
               line_thickness=8)
             return self.image_np
## <---------------------- Mail Sending ------------------------>


class Mail:
    def __init__(self):
        self.User_Detail()
        self.Receiver_detail()

    def User_Detail(self):
        self.From = "jawaharlakshmanan6@gmail.com"
        self.password = "BorntowiN"

    def Receiver_detail(self):
        self.To = ["jaiaravindh111399@gmail.com","thomassjerin3@gmail.com"]
        self.to = "jaiaravindh111399@gmail.com"

    def read_image(self):
        self.file_name = "1.jpg"
        self.image_data = open(self.file_name,"rb").read()
        return self.image_data

    def Message_Creation(self):
        msg = MIMEMultipart()
        msg["From"] = self.From
        msg["To"] = self.to
        msg["subject"] =  'subject : Alert!! '+self.name+' Animal found...'
        text = MIMEText("test")
        return text, msg

    def Message_Passing(self,name):
        self.name = name
        text, msg = self.Message_Creation()
        msg.attach(text)
        # image getting from read_image
        img_data = self.read_image()
        image = MIMEImage(img_data, name=os.path.basename(self.file_name))
        msg.attach(image)
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(self.From, self.password)
        s.sendmail(self.From, self.To, msg.as_string())
        s.quit()
        print ('successfully sent the mail')



## <----------------------- opencv ------------------------------>

import cv2
import time

class VideoCamera:
    def __init__(self):
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 18000)
        self.vid.set(cv2.CAP_PROP_FPS, 25)
        time_counter = 2
        seconds = 0
        while True:
            if time_counter == 3:
                if second == time.localtime()[5]:
                    time_counter = 2
                else:
                    pass

            self.ret, self.frame = self.vid.read()
            if self.ret:
                try :
                    self.frame, self.cls = tf1.show_inference(tf1.detection_model, self.frame)
                    # DISPLAYING THE BOUND OF PROPABALILITY
                    elephant_count = 0
                    ## timer for 30 seconds
                    if time_counter == 2:

                        for i in self.cls:
                            if (i.split()[0][:-1] in ["elephant","zebra"]):#'cat'or'dog'or'horse'or'sheep'or'cow'or'elephant'or'bear'or'zebra'or'giraffe'):
                                if float(i.split()[1][:-1]) >= 75.0:
                                    print("we found elephant please call the forest ranger")
                                    print("elephant count : ", self.cls)
                                    cv2.imwrite("1.jpg", self.frame)
                                    cv2.imshow("resulted picture ",self.frame)
                        #            playsound('aud2.mpeg')
                                    m1.Message_Passing(i.split()[0][:-1])
                         #           port.write(str.encode('A'))
                                    ## --> setting Timer
                                    second = time.localtime()[5]
                                    second = second + 30
                                    second = second % 60
                                    time_counter = 3

                                else:
                                    print("something found like an elephant")

                except:
                    self.frame = tf1.show_inference(tf1.detection_model, self.frame)

                cv2.imshow("fds", self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()

    def frame_capture(self):
        pass



if __name__ == "__main__":
    print("starting.......")
    tf1 = Tensorflow_mobile_SSD()
    m1 = Mail()
    v1 = VideoCamera()
    print("finished")
