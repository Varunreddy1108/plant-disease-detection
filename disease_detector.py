import os
import cv2
from tensorflow.keras.models import load_model

h=112
w=112
c = 3
cnn_model = load_model("model.hdf5")

ref = {0 : {"Apple (Cedar apple rust)":r"https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/"},
       1 : {"Apple healthy":"The plant leaf is healthy, Keep it up!"},
       2 : {"Corn (maize) Cercospora Gray leaf spot":r"https://extension.umn.edu/corn-pest-management/gray-leaf-spot-corn#:~:text=Gray%20leaf%20spot%20is%20typically,high%20humidity%20and%20warm%20conditions."},
       3 : {"Corn (maize) healthy":"The plant leaf is healthy, Keep it up!"},
       4 : {"Grape (Black_Measles)":r"https://www2.ipm.ucanr.edu/agriculture/grape/esca-black-measles/"},
       5 : {"Grape healthy":"The plant leaf is healthy, Keep it up!"},
       6 : {"Peach Bacterial spot":r"https://www.canr.msu.edu/news/management_of_bacterial_spot_on_peaches_and_nectarines#:~:text=Bacterial%20spot%20is%20an%20important,leaf%20spots%2C%20and%20twig%20cankers."},
       7 : {"Peach healthy":"The plant leaf is healthy, Keep it up!"},
       8 : {"Potato Early blight":r"https://www2.ipm.ucanr.edu/agriculture/potato/Early-Blight/"},
       9 : {"Potato healthy":"The plant leaf is healthy, Keep it up!"}}
  
def predictor(img_path):
    im = cv2.imread(img_path)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    if img.shape[0] > img.shape[1]:
        print("rotate")
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.resize(img,(h,w))
    img = img/255
    img = img.reshape(1,h,w,c)
    out = cnn_model.predict(img)
    res = ref[out.argmax()]
    perc = (out[0][out.argmax()]) * 100
    return res,perc,
