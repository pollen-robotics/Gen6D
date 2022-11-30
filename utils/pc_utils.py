import numpy as np
import pcl
from scipy.spatial.transform import Rotation as R

cube_sample = []
for i in range(-5, 20):
    for j in range(-5, 10):
        for k in range(-5, 30):
            cube_sample.append((i*0.01, j*0.01, k*0.01))

cube_sample = np.array(cube_sample)

def generatePlane(coeffs, atol=0.005):

    coeffs = np.around(coeffs, 5)
    a = coeffs[0]
    b = coeffs[1]
    c = coeffs[2]
    d = coeffs[3]

    c0 = np.array([a, b, c])
    p = cube_sample

    p_in_plane = p[np.isclose(p.dot(c0) + d, 0, atol=atol)]

    return p_in_plane

def segmentPC(cloud, planeTol=0.005, distanceThreshold = 0.025):    

    seg = cloud.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(distanceThreshold)
    seg.set_normal_distance_weight(0.01)
    seg.set_max_iterations(100)
    indices, coefficients = seg.segment()
    
    mask = np.ones(cloud.size, dtype=bool)
    mask[indices] = False
    objects = np.asarray(cloud)[mask]

    ground = generatePlane(coefficients, atol=planeTol)

    return ground, objects

def clipPoints(cloud, x=None, y=None, z=None):

    if x is not None:
        cloud = clipPointsAxis(cloud, 'x', x[0], x[1])
    if y is not None:
        cloud = clipPointsAxis(cloud, 'y', y[0], y[1])
    if z is not None:
        cloud = clipPointsAxis(cloud, 'z', z[0], z[1])

    return cloud

def rotatePoints(points:list, rotation:list, center:list=[0, 0, 0], degrees:bool=True):
    pp = []
    rot_mat = R.from_euler('xyz', rotation, degrees=degrees).as_matrix()
    for point in points:
        pp.append(rot_mat @ point)

    return pp


def translatePoints(points:list, translation):
    pp = []
    for point in points:
        pp.append(point + translation)

    return pp

def clipPointsBox(cloud, rot, trans, xRange, yRange, zRange):
    out = pcl.PointCloud()
    clipper = cloud.make_cropbox()
    # clipper.set_Translation(trans[0], trans[1], trans[2])
    # clipper.set_Rotation(rot[0], rot[1], rot[2])
    clipper.set_MinMax(xRange[0], yRange[0], zRange[0], 0, xRange[1], yRange[1], zRange[1], 0)
    out = clipper.filter()

    return out


def clipPointsAxis(cloud, axis, min, max):
    passthrough = cloud.make_passthrough_filter()
    passthrough.set_filter_field_name(axis)
    passthrough.set_filter_limits(min, max)
    cloud = passthrough.filter()

    return cloud

def downsample(cloud, leaf_size=0.03):
    vgf = cloud.make_voxel_grid_filter()
    vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
    cloud = vgf.filter()

    return cloud

def getClusterIndices(cloud, clusterTol=0.01, minClusterSize=200, maxClusterSize=25000):
    tree = cloud.make_kdtree()
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance (clusterTol)
    ec.set_MinClusterSize (minClusterSize)
    ec.set_MaxClusterSize (maxClusterSize)
    ec.set_SearchMethod (tree)
    cluster_indices = ec.Extract()

    return np.array(cluster_indices)

def getMiddleSlice(points, w, h):
    points              = points.reshape(h, w, 3)
    middle_slice        = points[:, w//2:w//2+1, :]
    middle_slice_points = middle_slice.reshape(h, 3)

    return middle_slice_points

def savePC(points, path):
    cloud = pcl.PointCloud()
    cloud.from_array(points)
    pcl.save(cloud, path)