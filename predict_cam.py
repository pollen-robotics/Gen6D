import argparse
import numpy as np
from tqdm import tqdm
from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
import sys
sys.path.insert(0, "./utils/")
from base_utils import load_cfg, project_points
from draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from pose_utils import pnp
import cv2
from realsense_wrapper import RealsenseWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
parser.add_argument('--database', type=str, default="custom/mouse")
# parser.add_argument('--output', type=str, default="data/custom/mouse/test")

# input video process
# parser.add_argument('--video', type=str, default="data/custom/video/mouse-test.mp4")
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

cfg = load_cfg(args.cfg)
ref_database = parse_database_name(args.database)
estimator = name2estimator[cfg['type']](cfg)
estimator.build(ref_database, split_type='all')

object_pts = get_ref_point_cloud(ref_database)
object_bbox_3d = pts_range_to_bbox_pts(np.max(object_pts,0), np.min(object_pts,0))

def get_pose_img(im, pose_init, hist_pts):
    h, w = im.shape[:2]
    f=np.sqrt(h**2+w**2)
    K = np.asarray([[f,0,w/2],[0,f,h/2],[0,0,1]],np.float32)
    if pose_init is not None:
        estimator.cfg['refine_iter'] = 1 # we only refine one time after initialization
    pose_pr, inter_results = estimator.predict(im, K, pose_init=pose_init)
    pose_init = pose_pr

    pts, _ = project_points(object_bbox_3d, pose_pr, K)
    bbox_img = draw_bbox_3d(im, pts, (0,0,255))

    hist_pts.append(pts)
    pts_ = weighted_pts(hist_pts, weight_num=args.num, std_inv=args.std)
    pose_ = pnp(object_bbox_3d, pts_, K)
    pts__, _ = project_points(object_bbox_3d, pose_, K)
    bbox_img_ = draw_bbox_3d(im, pts__, (0,0,255))

    return bbox_img_

pose_init = None
hist_pts = []
while True:
    
    # get and resize image
    im = rw.get_color_frame()
    cv2.imshow("raw im", im)
    # h, w = im.shape[:2]
    # ratio = args.resolution/max(h,w)
    # ht, wt = int(ratio*h), int(ratio*w)
    # im = cv2.resize(im, (ht,wt), interpolation=cv2.INTER_LINEAR)
    pose_im = get_pose_img(im, pose_init, hist_pts)
    cv2.imshow("pose", pose_im)
    cv2.waitKey(1)




