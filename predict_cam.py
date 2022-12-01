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

fv = FramesViewer()
fv.start()

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

bbox_center_pose        = np.eye(4)
bbox_center_pose[:3, 3] = bbox_center

fv.createPointsList("bbox", object_bbox_3d, color=(0, 1, 0), size=10)
fv.createPointsList("BBOX_POINT", color=(1, 0, 1), size=15)

# Y'a un truc chelou avec la bbox
#   elle a une position un peu random (surement les coordonnées dans le point cloud que j'ai détouré)
#   quand l'affiche avec project_points() et draw_bbox_3d() ça marche nickel
#   par contre la pose calculée par get_pose a un offset en translation (qui semble être l'offset du centre de la BB par rapport à l'origine)
# Si je fais :
#   object_bbox_3d -= bbox_center
# Alors la BB a le même offset que la box, et la frame projetée dans l'image est nickel au centre de la BB
# Par contre, je peux pas juste translater à posteriori la frame de la valeur du centre de la BB, il doit y avoir un rescaling quelque part
# 
# Dans project_points() y'a ça qui est fait
#   Pourquoi ce RT[:, 3] en plus ? c'est surement la qu'on bouge les points de la BB pour qu'ils s'alignent avec l'objet mais je comprend pas bien
#       pts = pts @ RT[:3, :3].T + RT[:, 3]
#       pts = pts @ K.T

# object_bbox_3d -= bbox_center
# bbox_length = np.linalg.norm(object_bbox_3d[0] - object_bbox_3d[4]) # cylindre
# bbox_length = np.linalg.norm(object_bbox_3d[5] - object_bbox_3d[6]) # pince
# factor = ref_database.size_meters[0]/bbox_length

factor = ref_database.size_meters[0]/np.linalg.norm(ref_database.x) # ça a l'air de marcher, attendre de bien mesurer avant de virer le truc avec la taille de la BB

arucoUtils = ArucoUtils(8, 5, 0.03, 0.015, aruco.DICT_4X4_50, (0.15, 0.24), "configs/calibration.pckl")

h, w = (480, 640)
f    = np.sqrt(h**2 + w**2)
K    = np.asarray([[f, 0, w/2], [0 ,f ,h/2], [0 ,0 ,1]], np.float32)

def get_pose(im, pose_init):

    if pose_init is not None:
        estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization

    # TODO should work better but does not -> investigate
    # pose_pr, _             = estimator.predict(im, arucoUtils.getCameraMatrix(), pose_init=pose_init)
    pose_pr, _             = estimator.predict(im, K, pose_init=pose_init)

    pose                   = np.vstack((pose_pr, [0, 0, 0, 1]))

    return pose

def get_bbox_img(im, pose):
    pts, _   = project_points(object_bbox_3d, pose, K)
    bbox_img = draw_bbox_3d(im, pts, (0,0,255))

    return bbox_img


pose_init = None
hist_pts = []
i = 0
while True:

    # When pose_init is none, the 4 steps are computed (detection, selection, pose, refine)
    # when it is not, juste the refine step is computed, initialized with the previous pose_init
    if i%20==0:
        pose_init = None

    im = rw.get_color_frame()

    T_camera_object = get_pose(im, pose_init)
    pose_init       = T_camera_object.copy()

    pose_im         = get_bbox_img(im, T_camera_object.copy())

    pt_center = bbox_center
    RT  = T_camera_object[:3, :]
    pt_center = pt_center @ RT[:3, :3].T + RT[:, 3]
    pt_center = pt_center @ K.T

    T_camera_bbox_center        = np.eye(4)
    T_camera_bbox_center[:3, 3] = pt_center

    T_world_camera = arucoUtils.get_camera_pose(im)
    if T_world_camera is not None:
        fv.pushFrame(T_world_camera, "T_world_camera")

        # T_world_bbox  = T_world_camera @ T_camera_bbox
        # T_world_bbox *= factor

        # fv.pushFrame(T_world_bbox, "T_world_bbox")

        T_world_camera = np.linalg.inv(T_world_camera)

        # pose_im = arucoUtils.drawFrame(pose_im, T_world_camera, length=0.2)

    pose_im = arucoUtils.drawFrame(pose_im, T_camera_bbox_center)

    cv2.imshow("pose", pose_im)
    cv2.waitKey(1)

    i += 1

    time.sleep(0.01)




