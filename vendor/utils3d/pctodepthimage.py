import open3d as o3d
import numpy as np
from PIL import Image


def pctodepthimage(path: str,
                  extrinsics: np.ndarray,
                  intrinsics: np.ndarray,
                  height: int,
                  width: int,
                  scaling_factor: int = 0.15) -> Image:
    """
    Convert point clouds to a depth image.
    Returns a B&W image.
    
    Parameters
    ----------
    path: str
        path to your point cloud file
    extrinsics: np.ndarray
        3x4 matrix containing the extrinsic parameters.
    intrinsics: np.ndarray
        3x3 matrix containing the intrinsic parameters.
    height: int
        height of the depth image.
    width: int
        width of the depth image.
    scaling_factor: int
        scaling factor decides the intensity of the pixel in the image. (defaults to 0.15)
        
    Returns
    -------
    PIL.Image
        depth image created after tranforming the point cloud
        
    Examples
    --------
    >>> from utils3d.pctodepthimage import pctodepthimage
    >>> depth_image = pctodepthimage(path, extrinsics, intrinsics, height, width, 0.15)
    >>> depth_image.show()
    """
    
    pcd = o3d.io.read_point_cloud(path)
    depth_image = np.zeros((height, width), dtype=np.uint8)
    
    """
    Creating the camera matrix by multiplying the intrinsic and extrinsic matrices.
    Results in a 3x3 matrix.
    """
    camera_matrix = np.matmul(intrinsics, extrinsics)
    
    for point in pcd.points:
        """ Reshaping the 3D point to (4, 1) vector by appending 1 """
        reshaped_point = np.append(np.asarray(point), 1).reshape(4, 1)
        
        """ Multiplying the camera matrix and the reshaped point vector to get the image coordinates (x, y, depth). """
        image_points = np.matmul(camera_matrix, reshaped_point)
        
        z = image_points[2]
        x = int(image_points[0]/z)
        y = int(image_points[1]/z)
        
        if (y > 0 and y < height - 1 and x > 0 and x < width - 1 and z > 0):
            depth_image[y][x] = 255 * int(z) * scaling_factor
            
    depth_image = Image.fromarray(depth_image)
    return depth_image
