import numpy as np
import cv2
class Regression:
    n = 5
    pointX = []
    pointY = []
    timeStep = list(range(1, n+1))

    t11 = n
    t12 = np.sum(timeStep)
    t13 = np.sum(np.power(timeStep, 2))

    t21 = t12
    t22 = t13
    t23 = np.sum(np.power(timeStep, 3))

    t31 = t22
    t32 = t23
    t33 = np.sum(np.power(timeStep, 4))

    T = np.array([[t11, t12, t13],
                  [t21, t22, t23],
                  [t31, t32, t33]])
    Bx = np.linalg.inv(T.T @ T) @ T.T
    By = np.linalg.inv(T.T @ T) @ T.T
    def predict(self, x, y, img):
        if len(self.pointX)  < self.n:
            self.pointX.append(x)
            self.pointY.append(y)
            return 0,0



        if len(self.pointX) == self.n:
            self.pointX.pop(0)
            self.pointY.pop(0)

        self.pointX.append(x)
        self.pointY.append(y)


        x11 = sum(self.pointX)
        x12 = sum(self.timeStep * np.array(self.pointX))
        x13 = sum(np.power(self.timeStep,2) * np.array(self.pointX))

        y11 = sum(self.pointY)
        y12 = sum(self.timeStep * np.array(self.pointY))
        y13 = sum(np.power(self.timeStep,2) * np.array(self.pointY))

        X = np.array([x11,
                      x12,
                      x13]).reshape((3,1))

        Y = np.array([y11,
                      y12,
                      y13]).reshape((3, 1))


        Bx = self.Bx @ X
        By = self.By @ Y
        self.img = img
        for a in range(10):
            x_predict = Bx[0] + Bx[1]*(a) + Bx[2]*(a)**2
            y_predict = By[0] + By[1] *(a) + By[2] * (a)**2
            cv2.circle(img, (int(x_predict), int(y_predict)), 3, (25*a, 255, 25*a), 2)

        return int(x_predict), int(y_predict)





