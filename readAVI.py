import cv2
import os
from PIL import Image
from pylab import *

# path = r'E:/mywork/6月/frame/'
cap = cv2.VideoCapture('C:/Users/64426/Desktop/seuGraph Project/ball_line/单人投篮 00_02_41-00_02_44_out.avi')
i = 2
# if not os.path.exists(path):
#     os.makedirs(path)

if cap.isOpened():
    ret, frame = cap.read()  # ret是bool型，当读完最后一帧就是False，frame是ndarray型
    # if not ret:  #
    #     break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #
    # cv2.imwrote(' ',gray) #路径无中文存图
    cv2.imencode('.jpg', gray)[1].tofile('./image6/frame_' + str(i) + '.jpg')  # 路径含中文存图
    imshow(gray)
    i += 1

short_pointUV = ginput(1)
short_pointUV = np.array(short_pointUV, dtype=np.double)
print("short_pointUV: ", short_pointUV)

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
