import open3d as o3d
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='log/sample/fps_0_8p_sample.xyzrgb')
FLAGS = parser.parse_args()

FLIE_PATH = FLAGS.file_path

pcd = o3d.io.read_point_cloud(FLIE_PATH)
o3d.visualization.draw_geometries([pcd], width=720, height=480)