import cv2
from kalmanfilter import KalmanFilter
from kalmanfilter import KF
import numpy as np

kfnew = KF()
kf = KalmanFilter()
cap = cv2.VideoCapture("Test.mp4")



tracker = cv2.legacy.TrackerMOSSE_create()
success, img = cap.read()
bbox = cv2.selectROI("Tracking", img, False)
tracker.init(img, bbox)

def onMouse(event, x,y, flags, param):
    print('x = %d, y = %d' % (x, y))
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]),int(bbox[2]),int(bbox[3])
    cv2.rectangle(img,(x,y),((x + w), (y+h)), (255,0,255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(bbox)
def ringBuffer(bbox):
    buf = np.zeros((10,2), dtype = int)

while True:
    ret, img = cap.read()

    timer = cv2.getTickCount()
    success, img = cap.read()
    success, bbox = tracker.update(img)
    print(bbox)
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

    predicted = kf.predict(centerX, centerY)

    cv2.circle(img, (centerX, centerY), 5, (255, 255, 255), 4)#центр цели


    predicted = kf.predict(predicted[0], predicted[1])
    cv2.circle(img, predicted, 5, (0, 255, 255), 4)  # старый калман

    predictedNewCalman = kfnew.predict(centerX, centerY)
    cv2.circle(img,  predictedNewCalman, 5, (255, 0, 0), 4) #новый калман





    cv2.imshow("Frame", img)
    cv2.setMouseCallback("Frame", onMouse)
    key= cv2.waitKey(0
                     )
    if key ==27:
        break