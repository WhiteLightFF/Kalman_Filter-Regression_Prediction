import numpy as np
class Regression:

    def predict(self, pointX, pointY, n):
        timeStep = list(range(1,n+1))
        #timeStep = [1, 3, 4, 7, 9]


        t11 = n
        t12 = np.sum(timeStep)
        t13 = np.sum(np.power(timeStep, 2))

        t21 = t12
        t22 = t13
        t23 = np.sum(np.power(timeStep, 3))

        t31 = t22
        t32 = t23
        t33 = np.sum(np.power(timeStep, 4))


        T = np.array( [[t11, t12, t13],
                       [t21, t22, t23],
                       [t31, t32, t33]])

        self.pointX = pointX

        x11 = sum(pointX)
        x12 = sum(timeStep * np.array(pointX))
        x13 = sum(np.power(timeStep,2) * np.array(pointX))

        y11 = sum(pointY)
        y12 = sum(timeStep * np.array(pointY))
        y13 = sum(np.power(timeStep,2) * np.array(pointY))

        X = np.array([x11, x12, x13]).reshape((3,1))
        Y = np.array([y11, y12, y13]).reshape((3, 1))


        Bx = np.linalg.inv(T.T @ T) @ T.T @ X
        By = np.linalg.inv(T.T @ T) @ T.T @ Y

        x_predict = Bx[0] + Bx[1]*(n+3) + Bx[2]*(n+3)**2
        y_predict = By[0] + By[1] *(n+3) + By[2] * (n+3)**2
        return np.round(x_predict), np.round(y_predict)





lstX = [837, 835, 836, 840, 847]
lstY = [223, 240, 255, 274, 291]
reg = Regression()
print(reg.predict(lstX, lstY, 5))