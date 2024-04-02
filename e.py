# code based on
# very very good, very good, and compare vith tesla
# https://medium.com/analytics-vidhya/depth-sensing-and-3d-reconstruction-512ed121aa60

# other good links
# https://www.youtube.com/watch?v=k_QSqbj_bYo


import cv2
import numpy as np
import open3d as o3d
import os

# Load stereo images
current_directory = os.path.dirname(os.path.abspath(__file__))

# Load stereo images
img_left =cv2.imread( os.path.join(current_directory, 'L.jpg'))
img_right = cv2.imread( os.path.join(current_directory, 'R.jpg'))

# Convert images to grayscale
img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# Create stereoBM object
#stereo = cv2.StereoBM_create(numDisparities=96, blockSize=11)
stereo = cv2.StereoSGBM_create(
    minDisparity=12,
    numDisparities=64, blockSize=21)

# Compute disparity
disparity = stereo.compute(img_left_gray, img_right_gray)

# Normalize disparity for visualization
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Show the disparity map
#cv2.imshow('Disparity Map', disparity)
#cv2.waitKey(0)
cv2.imshow('Disparity Map', disparity_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

#region Rendering usign open3D


# Create point cloud from disparity map
h, w = img_left_gray.shape[:2]
focal_length = 0.8 * w
Q = np.float32([[1, 0, 0, -0.5 * w],
                [0, -1, 0, 0.5 * h],
                [0, 0, 0, -focal_length],
                [0, 0, 1, 0]])

points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Mask out invalid points
mask = disparity > disparity.min()
points_3D = points_3D[mask]
colors = img_left[mask]

# Adjust scaling factor
scaling_factor = 1.0 / 16
points_3D *= scaling_factor

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3D)
pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

print("Points:", points_3D.shape)
#print("Colors:", colors.shape)
print(pcd)

# Visualize the point cloud
# o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd], window_name='Open3D', width=800, height=600)


#endregion  