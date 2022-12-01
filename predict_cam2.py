import argparse
import numpy as np
from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from utils.base_utils import load_cfg, project_points, project_points2
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp
from utils.realsense_wrapper import RealsenseWrapper
from utils.aruco_utils import ArucoUtils
from cv2 import aruco
import cv2
from FramesViewer import FramesViewer
import time
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
parser.add_argument('--database', type=str, default="custom/mouse")
parser.add_argument('--resolution', type=int, default=960)
parser.add_argument('--transpose', action='store_true', dest='transpose', default=False)
# smooth poses
parser.add_argument('--num', type=int, default=5)
parser.add_argument('--std', type=float, default=2.5)
args = parser.parse_args()

rw = RealsenseWrapper()

def weighted_pts(pts_list, weight_num=10, std_inv=10):
    weights=np.exp(-(np.arange(weight_num)/std_inv)**2)[::-1] # wn
    pose_num=len(pts_list)
    if pose_num<weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:,None,None],0)/np.sum(weights)
    return pts

cfg          = load_cfg(args.cfg)
ref_database = parse_database_name(args.database)
estimator    = name2estimator[cfg['type']](cfg)
estimator.build(ref_database, split_type='all')

object_pts       = get_ref_point_cloud(ref_database)
object_bbox_3d   = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))
bbox_center      = np.mean(object_bbox_3d, axis=0)
bbox_rotation    = ref_database.rotation

# Il y a une rotation pas claire encore. Je pensais que c'était celle de ref_database.rotation mais apparemment non (?)
# sur le cylindre, on peut mettre -np;pi/2 sur en y, mais c'est pas bon pour la pince
bbox_rotation    = R.as_matrix(R.from_euler("xyz", [0, -np.pi/2, 0]))
# bbox_rotation    = R.as_matrix(R.from_euler("xyz", [0, 0, 0]))

initial_pose         = np.eye(4)
initial_pose[:3, :3] = bbox_rotation
initial_pose[:3, 3]  = bbox_center


len_x           = np.linalg.norm(object_bbox_3d[0] - object_bbox_3d[4])
len_y           = np.linalg.norm(object_bbox_3d[7] - object_bbox_3d[4])
len_z           = np.linalg.norm(object_bbox_3d[3] - object_bbox_3d[0])

bbox_dimensions = [len_x, len_y, len_z]

factor = ref_database.size_meters[0]/np.linalg.norm(ref_database.x)
# bbox_dimensions = ref_database.size_meters/factor
arucoUtils = ArucoUtils(8, 5, 0.03, 0.015, aruco.DICT_4X4_50, (0.15, 0.24), "configs/calibration.pckl")

h, w = (480, 640)
f    = np.sqrt(h**2 + w**2)
K    = np.asarray([
                    [f, 0, w/2], 
                    [0, f, h/2], 
                    [0, 0, 1  ]
                ],np.float32)


def get_pose(im, pose_init):

    if pose_init is not None:
        estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization

    pose_pr, _             = estimator.predict(im, K, pose_init=pose_init)
    pose                   = np.vstack((pose_pr, [0, 0, 0, 1]))

    return pose

# getBBoxPoints(pose, dimensions)

def get_bbox_img(im, pose):
    pts, _   = project_points(object_bbox_3d, pose, K)
    bbox_img = draw_bbox_3d(im, pts, (0,0,255))

    return bbox_img

pose_init = None
i         = 0

while True:

    if i%20 == 0:
        pose_init = None


    im              = rw.get_color_frame()

    T_camera_object = get_pose(im, pose_init)
    pose_init       = T_camera_object.copy()

    pose_im         = get_bbox_img(im, T_camera_object)
    pose_im         = arucoUtils.drawFrame(pose_im, T_camera_object @ initial_pose, length=1)
    pose_im         = arucoUtils.drawBBox(pose_im, object_bbox_3d, T_camera_object @ initial_pose, bbox_dimensions)

    cv2.imshow("pose", pose_im)
    cv2.waitKey(1)

    i += 1