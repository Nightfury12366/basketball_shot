import cv2
import os

# path = r'E:/mywork/6月/frame/'
cap = cv2.VideoCapture('C:/Users/64426/Desktop/ball_line/单人投篮 00_00_34-00_00_37_out.avi')
i = 1
# if not os.path.exists(path):
#     os.makedirs(path)

if cap.isOpened():
    ret, frame = cap.read()  # ret是bool型，当读完最后一帧就是False，frame是ndarray型
    # if not ret:  #
    #     break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  #
    # cv2.imwrote(' ',gray) #路径无中文存图
    cv2.imencode('.jpg', gray)[1].tofile('./image6/frame_' + str(i) + '.jpg')  # 路径含中文存图
    cv2.imshow('frame', gray)
    i += 1

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
