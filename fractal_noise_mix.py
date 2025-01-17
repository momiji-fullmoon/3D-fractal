import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def mix_point_clouds_with_colors(pc1, pc2, ratio=0.5):
    """
    Mix two point clouds with colors to differentiate their origin.
    :param pc1: First point cloud (N x 3 numpy array)
    :param pc2: Second point cloud (M x 3 numpy array)
    :param ratio: Ratio of points from the first point cloud in the mixed result
    :return: Mixed point cloud with labels (P x 3 numpy array, P x 1 array for labels)
    """

    # Calculate the number of points to sample from each point cloud.
    num_pc1 = int(len(pc1) * ratio)
    # Change: Ensure num_pc2 is not larger than the size of pc2
    num_pc2 = int(len(pc2) * (1-ratio))  
    # Previously: num_pc2 = len(pc1) + len(pc2) - num_pc1
    # This would make num_pc2 larger than len(pc2) if ratio<0.5
    # Randomly sample points from pc1 and pc2
    pc1_sampled = pc1[np.random.choice(pc1.shape[0], num_pc1, replace=False)]
    pc2_sampled = pc2[np.random.choice(pc2.shape[0], num_pc2, replace=False)]

    # Combine points and labels
    mixed_points = np.vstack([pc1_sampled, pc2_sampled])
    labels = np.hstack([np.zeros(num_pc1), np.ones(num_pc2)])  # 0: pc1, 1: pc2

    return mixed_points, labels

def visualize_mixed_point_cloud_matplotlib(mixed_pc, labels):
    """
    Visualize the mixed point cloud with Matplotlib.
    :param mixed_pc: Mixed point cloud (P x 3 numpy array)
    :param labels: Labels indicating the origin of each point (P x 1 numpy array)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Separate points by labels
    pc1_points = mixed_pc[labels == 0]
    pc2_points = mixed_pc[labels == 1]

    # Plot point cloud 1 (red) and point cloud 2 (green)
    ax.scatter(pc1_points[:, 0], pc1_points[:, 1], pc1_points[:, 2], c='r', label='Point Cloud 1', alpha=0.6)
    ax.scatter(pc2_points[:, 0], pc2_points[:, 1], pc2_points[:, 2], c='g', label='Point Cloud 2', alpha=0.6)

    ax.set_title("Mixed Point Clouds (Colored by Origin)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load point clouds from PLY files
    pc1_file = "point1.ply"
    pc2_file = "point2.ply"

    pc1_o3d = o3d.io.read_point_cloud(pc1_file)
    pc2_o3d = o3d.io.read_point_cloud(pc2_file)

    # Convert Open3D point clouds to numpy arrays
    pc1 = np.asarray(pc1_o3d.points)
    pc2 = np.asarray(pc2_o3d.points)

    # Mix the point clouds with a 50-50 ratio and assign labels
    mixed_pc, labels = mix_point_clouds_with_colors(pc1, pc2, ratio=0.5)

    # Visualize the mixed point cloud with Matplotlib
    visualize_mixed_point_cloud_matplotlib(mixed_pc, labels)
