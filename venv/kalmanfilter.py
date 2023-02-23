#https://pysource.com/2021/10/29/kalman-filter-predict-the-trajectory-of-an-object/
import cv2
import numpy as np


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


class KF:
    A = np.array([[1,0],
                  [0,1]])
    B = np.array([[1, 0],
                  [0, 1]])
    Q = np.array([[0.3, 0],
                  [0, 0.3]])#по сути ускорение
    P = np.array([[0.01,0],
                  [0,0.01]]) #Матрица ковариации
    R = np.array([[0.75,0],
                  [0,0.6]])
    H = np.array([[1, 0], 
                  [0, 1]])
    point_past= np.array([0, 0]).reshape((2, 1))
    point_current = np.array([0., 0.]).reshape((2, 1))
    velocity = np.array([1, 1]).reshape((2, 1))
    def predict(self, coordX, coordY):
        #prediction
        point_predict = self.A @ self.point_past + self.B @ self.velocity
        P_predict = self.A @ self.P @ self.A.T + self.Q
        #update
        self.point_current[0] = coordX
        self.point_current[1] = coordY
        sigmaPoint = self.point_current - self.H @ point_predict
        sigmaP = self.H @ P_predict @ self.H.T + self.R
        K = P_predict @ self.H.T @ np.linalg.inv(sigmaP)
        #update variables
        self.point_past = point_predict + K @ sigmaPoint
        self.P = P_predict - K @ self.H @ P_predict
        x, y = int(self.point_past[0]), int(self.point_past[1])
        return x,y
