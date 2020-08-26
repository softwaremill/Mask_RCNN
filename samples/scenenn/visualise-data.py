import numpy as np
from open3d import *
from plyfile import PlyData
from matplotlib import pyplot as plt

ROOT_DIR = os.path.abspath("../../")
DATA_DIR = os.path.join(ROOT_DIR, "data/scenenn_seg_76/scenenn_seg_76_raw/scenenn_dec24_data")

def main():
    plydata = PlyData.read(os.path.join(DATA_DIR, "005/005.ply"))
    labels = plydata['vertex']['label']

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cloud = io.read_point_cloud(os.path.join(DATA_DIR, "005/005.ply")) # Read the point cloud
    cloud.colors = utility.Vector3dVector(colors[:, :3])
    visualization.draw_geometries([cloud]) # Visualize the point cloud

if __name__ == "__main__":
    main()