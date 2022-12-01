import cv2
from cv2 import aruco
import pickle
import numpy as np

class ArucoUtils:
    def __init__(self, charuco_rows, charuco_cols, squareLength, markerLength, aruco_dict, board_size, calibration_file_path):
        self.__charucoRows = charuco_rows
        self.__charucoCols = charuco_cols
        self.__squareLength = squareLength
        self.__markerLength = markerLength
        self.__arucoDict    = aruco.Dictionary_get(aruco_dict)
        self.__boardSize   = board_size

        self.__cameraMatrix, self.__distCoeffs, _, _ = pickle.load(open(calibration_file_path, "rb"))

        self.__charucoBoard = aruco.CharucoBoard_create(
                                    squaresX     = self.__charucoCols,
                                    squaresY     = self.__charucoRows,
                                    squareLength = self.__squareLength,
                                    markerLength = self.__markerLength,
                                    dictionary   = self.__arucoDict
                                )

    def getCameraMatrix(self):
        return self.__cameraMatrix
                
    def get_camera_pose(self, im):

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(
                                    image      = gray,
                                    dictionary = self.__arucoDict
                                )
        if len(corners) > 0:

            aruco.refineDetectedMarkers(gray, self.__charucoBoard, corners, ids, rejected)

            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, self.__charucoBoard)

            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, self.__charucoBoard, self.__cameraMatrix, self.__distCoeffs, None, None)

            if retval == True:
                T_camera_board = np.eye(4)

                T_camera_board[:3, :3], _ = cv2.Rodrigues(rvec.flatten())
                T_camera_board[:3, 3] = tvec.flatten()

                T_board_camera = np.linalg.inv(T_camera_board)

                return T_board_camera

        return None

    def drawFrame(self, im, frame, length=3):
        h, w = im.shape[:2]
        f    = np.sqrt(h**2 + w**2)
        K    = np.asarray([[f, 0, w/2], [0 ,f ,h/2], [0 ,0 ,1]], np.float32)
        
        tvec = frame[:3, 3]
        rvec, _ = cv2.Rodrigues(frame[:3, :3])

        # return cv2.drawFrameAxes(im, self.__cameraMatrix, self.__distCoeffs, rvec, tvec, length=length)
        return cv2.drawFrameAxes(im, K, np.array([0., 0., 0., 0., 0.]), rvec, tvec, length=length)

    # VALIDÉ
    def getBBoxPoints(self, points, frame, dimensions):
        x = dimensions[0]
        y = dimensions[1]
        z = dimensions[2]

        points = np.array([
                    [ x/2,  y/2 , z/2],
                    [ x/2,  y/2, -z/2],
                    [ x/2, -y/2,  z/2],
                    [ x/2, -y/2, -z/2],
                    [-x/2,  y/2 , z/2],
                    [-x/2,  y/2, -z/2],
                    [-x/2, -y/2,  z/2],
                    [-x/2, -y/2, -z/2]
                ])

        points = points @ frame[:3, :3].T + frame[:3, 3]

        return points


    def drawBBox(self, im, pts, frame, dimensions):

        points = self.getBBoxPoints(pts, frame, dimensions)
        for point in points:
            f = np.eye(4)
            f[:3, 3] = point
            im = self.drawFrame(im, f, length=0.1)

        return im

        # TODO


