import cv2
from kalmanfilter import KalmanFilter
from kalmanfilter import KF
from regression import Regression
from LSTM import LSTM
import numpy as np
import csv


kfnew = KF()
cap = cv2.VideoCapture("Test2.mp4")
rg = Regression()
lstm = LSTM()

with open("Source.csv", "w") as file2:
    Source = csv.writer(file2)
    Source.writerow(("x0", "y0"))
with open("Predict.csv", "w") as file:
    Predict = csv.writer(file)
    Predict.writerow(("x1", "y1"))




#Tracker Init
tracker = cv2.legacy.TrackerMOSSE_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img, bbox)

def onMouse(event, x,y, flags, param):
    #print('x = %d, y = %d' % (x, y))
    pass
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x + w), (y+h)), (255,0,255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #print(bbox)
def ringBuffer(bbox):
    buf = np.zeros((10,2), dtype = int)

while True:
    ret, img = cap.read()
    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)
    if success:
        drawBox(img, bbox)
    else:
        cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    if ret is False:
        break
    centerX = int(bbox[0]+ bbox[2]/2)
    centerY = int(bbox[1]+ bbox[3]/2)
    coord = [centerX, centerY]
    print(centerX, centerY)
    with open("Source.csv", "a") as file2:
        Source = csv.writer(file2)
        Source.writerow(coord)

    cv2.circle(img, (centerX, centerY), 5, (255, 255, 255), 4)#center object
    #KalmanFilter
    predictedNewCalman = kfnew.predict(centerX, centerY)
    cv2.circle(img, predictedNewCalman, 5, (255, 0, 0), 4)
    #regression
    predictedReg = rg.predict(centerX, centerY, img)
    #LSTM
    # predictedLSTM = lstm.LSTM_Predict(centerX, centerY)
    # for i in range(0,len(predictedLSTM)):
    #     cv2.circle(img, (int(predictedLSTM[i][0]), int(predictedLSTM[i][1])), 5, (85*i, 85*i, 255), 4)






    cv2.imshow("Frame", img)
    cv2.setMouseCallback("Frame", onMouse)
    key= cv2.waitKey(0)
    if key ==27:
        break