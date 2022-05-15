import os
import cv2
import time
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import zipfile
import glob
import re
from pathlib import Path
import pandas as pd
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab import drive
from mtcnn.mtcnn import MTCNN
from numpy import asarray
import cv2
from PIL import Image
!pip install imagesize
import imagesize

class DataSetCreator:
  def __init__(self, dataset_path, image_dimensions, fresh_create=False):
    self.name = name
    self.age = age
	drive.mount('/content/gdrive')
    
  def extract_lfw_mlfw():
    if(os.path.isdir('/tmp/lfw') == False):
      cmd = "tar -xvf gdrive/MyDrive/datasets/lfw.tar -C /tmp/"
      returned_value = os.system(cmd)
    if(os.path.isdir('/tmp/lfw-deepfunneled') == True):
      os.rename('/tmp/lfw-deepfunneled','/tmp/lfw')

    if(os.path.isdir('/tmp/mlfw/')== False):
      zip_ref = zipfile.ZipFile("/content/gdrive/MyDrive/datasets/MLFW.zip", 'r')
      zip_ref.extractall("/tmp/mlfw/")
      zip_ref.close()
  
  def pick_images_lfw():
    lfw_images = []
    lfw_train = sorted(os.listdir("/tmp/lfw/"))
    for directory in lfw_train:
      dir_path = "/tmp/lfw/" + directory
      no_of_files = len(os.listdir(dir_path)) 
      if(no_of_files >= 3 and no_of_files <= 5):
        lfw_images.append(directory)
    return lfw_images;
  
  def create_common_dir_list():
    extract_lfw_mlfw();
    lfw_dir = pick_images_lfw();
    mlfw_dir = pick_images_mlfw()
    common_set = set(lfw_dir) & set(mlfw_dir)
    return list(common_set)
  
  def pick_images_mlfw():
    file_list = glob.glob("/tmp/mlfw/origin/*.jpg")
    final_name_list = []
    for file in file_list:
        file_name = Path(file).name
        final_str = ""
        tokens = file_name.split("_")
        for token in tokens:
            if(token.isnumeric()):
                break;
            else:
                final_str += token
                final_str += "_"
        final_name_list.append(final_str)  
    print(len(set(final_name_list)))
    df = pd.DataFrame(final_name_list)
    df.columns = ['name']
    new_df = df.groupby('name').filter(lambda x : (len(x) >= 3 and len(x) <= 5))
    mlfw_images = new_df.values.tolist()
    mlfw_images = [sub[0][ : -1] for sub in mlfw_images]
    return list(set(mlfw_images))
  
  def pick_classes(no_of_classes):
    final_list = []
    filtered_list =  create_common_dir_list()
    print("filtered_list_length = " + str(len(filtered_list)))
    for i in range(0, no_of_classes):
        final_list.append(random.choice(filtered_list))
    return final_list
  
  def create_train_test_set_from_lfw_mlfw():
    final_class_list = pick_classes(100)
    print(final_class_list)
    if(os.path.isdir('/tmp/dataset/train') == True):
      shutil.rmtree('/tmp/dataset/train')
    if(os.path.isdir('/tmp/dataset/test') == True):
      shutil.rmtree('/tmp/dataset/test')
    if(os.path.isdir('/tmp/dataset/')  == False):
      os.mkdir('/tmp/dataset/')
    if(os.path.isdir('/tmp/dataset/train')== False):
      os.mkdir('/tmp/dataset/train')
    if(os.path.isdir('/tmp/dataset/test')== False):
      os.mkdir('/tmp/dataset/test')

    for dir in final_class_list:  
      os.system("cp -r " + " /tmp/lfw/"+ dir + " /tmp/dataset/train")

    mlfw_jpg_list = []
    for path, subdirs, files in os.walk('/tmp/mlfw/aligned/'):
        for name in files:
            mlfw_jpg_list.append(os.path.join(path, name))

    for dir in final_class_list:
      if(os.path.isdir('/tmp/dataset/test/'+dir)  == False):
        os.mkdir('/tmp/dataset/test/'+dir)
      name_list = [k for k in mlfw_jpg_list if dir in k]
      for nme in name_list:
        shutil.copy(nme, "/tmp/dataset/test/"+dir + "/")
        
  def process_face(filename, required_size=(INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE)):
    # load image from file
    image = Image.open(filename)
    # # convert to RGB, if needed
    # image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if(results):
      # extract the bounding box from the first face
      x1, y1, width, height = results[0]['box']
      # bug fix
      x1, y1 = abs(x1), abs(y1)
      x2, y2 = x1 + width, y1 + height
      # extract the face
      face = pixels[y1:y2, x1:x2]
      # resize pixels to the model size
      image = Image.fromarray(face)
      image = image.resize(required_size)
      face_array = asarray(image)
      cv2.imwrite(filename, face_array)
      new_size = imagesize.get(filename)
      print("image size", new_size)
      if(new_size[0] != INPUT_IMAGE_SHAPE):
        print("deleted image size", filename,new_size)
        os.remove(filename)
        
  def get_image_list_from_folder(path):
      final_list = []
      for path, subdirs, files in os.walk(path):
        for name in files:
            final_list.append(os.path.join(path, name))
      return sorted(final_list)

  def process_faces(image_list):
  	  for file in image_list:
    	process_face(file)
        
  def plot_images(w, h, fig_w, fig_h, image_list, columns, rows):
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i in range(1, columns*rows +1):
        img = mpimg.imread(image_list[i])
        fig.add_subplot(rows, columns, i)
        plt.title(os.path.basename(image_list[i]))
        plt.imshow(img)
    plt.show()
    
    
  train_list = get_image_list_from_folder('/tmp/dataset/train/')
  process_faces(train_list)
  # train_len = int(len(train_list))
  # per_row = int(math.sqrt(train_len))
  subprocess.run(["%cd", "/content/gdrive/MyDrive/mfr_test_repo/MaskTheFace/"])
  %cd /content/gdrive/MyDrive/mfr_test_repo/MaskTheFace/
  !pwd
  # !pip install -r requirements.txt
  subprocess.run(["!pip", "install", "dotmap"])
  !pip install dotmap
  for path in Path('/tmp/dataset/train/').rglob('*.jpg'):
    print ("generating for path" , path)
    result = subprocess.run(["python", "/content/gdrive/MyDrive/mfr_test_repo/MaskTheFace/mask_the_face.py",'--path', path, '--mask_type', 'random', '--verbose', '--write_original_image'],  stdout=subprocess.PIPE, text=True)   
    print(result.stdout)
train_list = get_image_list_from_folder('/tmp/dataset/train/')
plot_images(10, 10, 50, 50, train_list, 5, 5)
print(train_list)

  