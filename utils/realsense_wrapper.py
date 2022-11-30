import pyrealsense2 as rs
import numpy as np
# from pc_utils import *
import cv2

class RealsenseWrapper():
    def __init__(self):

        self.pipeline    = rs.pipeline()
        self.config      = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)

        pipeline_profile.get_device()

    
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        other_stream, other_format = rs.stream.color, rs.format.rgb8
        self.config.enable_stream(other_stream, other_format, 30)

        self.pipeline.start(self.config)
        profile = self.pipeline.get_active_profile()

        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.color_intrinsics = color_profile.get_intrinsics()

        depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
        self.depth_intrinsics = depth_profile.get_intrinsics()
        self.w, self.h = self.depth_intrinsics.width, self.depth_intrinsics.height

        self.pc = rs.pointcloud()

    def get_depth_intrinsics(self):
        return self.depth_intrinsics

    def get_color_intrinsics(self):

        fx  = self.color_intrinsics.fx
        fy  = self.color_intrinsics.fy
        ppx = self.color_intrinsics.ppx
        ppy = self.color_intrinsics.ppy

        matrix = np.eye(3)
        matrix[0][0] = fx
        matrix[1][1] = fy
        matrix[0][2] = ppx
        matrix[1][2] = ppy

        matrix[0][1] = 0 # skew ?


        return matrix, np.array(self.color_intrinsics.coeffs)

    def get_color_frame(self):
        success, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
        if success:
            color_frame = np.asanyarray(frames.get_color_frame().get_data())
            return cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        else:
            return None

    # def get_raw_point_cloud(self):
    #     success, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
    #     if not success:
    #         return None

    #     depth_frame = frames.get_depth_frame().as_video_frame()
    #     points = np.array(self.pc.calculate(depth_frame).get_vertices(2))
        
    #     cloud = pcl.PointCloud()
    #     cloud.from_array(points)

    #     return cloud


    # TODO downsample only in to display, pcl operations should be done on full data
    # def get_point_cloud(self):
    #     success, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)

    #     if not success:
    #         return np.array([]), np.array([]), np.array([])

    #     depth_frame = frames.get_depth_frame().as_video_frame()
    #     points = np.array(self.pc.calculate(depth_frame).get_vertices(2))

    #     cloud = pcl.PointCloud()
    #     cloud.from_array(points)

    #     cloud = clipPoints(cloud, x=[-0.2, 0.2], y=[-0.2, 0.2], z=[0.1, 0.5])

    #     plane, objects = segmentPC(cloud, planeTol=0.002, distanceThreshold=0.005)

    #     cloud.from_array(objects)

    #     objects = downsample(cloud, leaf_size=0.002)

    #     clusterIndices = getClusterIndices(objects, minClusterSize=30)

    #     return np.array(plane), np.array(objects), clusterIndices