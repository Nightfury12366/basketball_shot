from PIL import Image
from pylab import *
import pickle
import numpy as np
import cv2

im_ = array(Image.open('./image6/raw.jpg'))
imshow(im_)
src_point = ginput(7)
object_2d_point = np.array(src_point, dtype=np.double)
with open('shot_line.pkl', 'wb') as out_data:
    pickle.dump(object_2d_point, out_data)

