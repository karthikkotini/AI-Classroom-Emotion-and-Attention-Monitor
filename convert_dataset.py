import pandas as pd
import numpy as np
import os
import cv2

# load csv
data = pd.read_csv("fer2013.csv")

emotion_map = {
    0:"angry",
    1:"disgust",
    2:"fear",
    3:"happy",
    4:"sad",
    5:"surprise",
    6:"neutral"
}

# create folders
for emotion in emotion_map.values():
    os.makedirs(f"dataset/{emotion}", exist_ok=True)

# convert pixels to images
for index,row in data.iterrows():

    emotion = emotion_map[row["emotion"]]
    pixels = np.array(row["pixels"].split(),dtype='uint8')
    image = pixels.reshape(48,48)

    filename = f"dataset/{emotion}/{index}.jpg"
    cv2.imwrite(filename,image)

print("Dataset conversion completed!")