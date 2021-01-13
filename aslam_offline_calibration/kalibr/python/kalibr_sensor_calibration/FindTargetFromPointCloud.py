import numpy as np
import open3d as o3d
from skimage.measure import LineModelND, ransac

from util import showPointCloud


def extract_plane_points(point_cloud):
    num_point_threshold = 20
    if point_cloud.shape[0] < num_point_threshold:
        return np.array([])
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    _, inliers = o3d_point_cloud.segment_plane(distance_threshold=0.03,
                                               ransac_n=num_point_threshold,
                                               num_iterations=100)
    return point_cloud[inliers]


def find_points_on_tapes(point_cloud):
    filtered_points = point_cloud[point_cloud[:, 4] > 200]
    filtered_points = extract_plane_points(filtered_points)
    return filtered_points


def fitting_tapes_lines(point_cloud):
    points_xyz = point_cloud[:, :3]
    num_inlier_threshold = max(int(points_xyz.shape[0] / 8), 5)
    try:
        tape1_model, tape1_inliers = ransac(points_xyz,
                                            LineModelND,
                                            min_samples=num_inlier_threshold,
                                            residual_threshold=0.03,
                                            max_trials=100)
    except Exception:
        return None, None

    rest_points_xyz = point_cloud[~tape1_inliers, :3]
    num_inlier_threshold = max(int(rest_points_xyz.shape[0] / 8), 5)
    try:
        tape2_model, tape2_inliers = ransac(rest_points_xyz,
                                            LineModelND,
                                            min_samples=num_inlier_threshold,
                                            residual_threshold=0.03,
                                            max_trials=100)
    except Exception:
        return None, None

    cosine_angle = np.dot(tape1_model.params[1], tape2_model.params[1])
    if abs(cosine_angle) > 0.087156:  # ~cos(85)
        return None, None

    return tape1_model.params, tape2_model.params


def estimate_intersection(line_params1, line_params2):
    p0 = line_params1[0]
    t0 = line_params1[1]
    p1 = line_params2[0]
    t1 = line_params2[1]
    a = np.array([[0.0, -t0[2], t0[1]],
                  [t0[2], 0.0, -t0[0]],
                  [0.0, -t1[2], t1[1]],
                  [t1[2], 0.0, -t1[0]]])
    b = np.array([t0[1] * p0[2] - t0[2] * p0[1],
                  t0[2] * p0[0] - t0[0] * p0[2],
                  t1[1] * p1[2] - t1[2] * p1[1],
                  t1[2] * p1[0] - t1[0] * p1[2]])
    estimated_intersection = np.linalg.lstsq(a, b, rcond=None)[0]
    return estimated_intersection


def estimate_rotation(axis_vector1, axis_vector2, intersection):
    z_axis = np.cross(axis_vector1, axis_vector2)
    if z_axis.dot(intersection) > 0:
        z_axis = z_axis * -1.0
        y_axis = axis_vector1
    else:
        y_axis = axis_vector2

    x_axis = np.cross(y_axis, z_axis)
    rotate_mat = np.column_stack([x_axis, y_axis, z_axis])
    return rotate_mat


def calculate_direction(line_centroid, intersection):
    epsilon = 0.1
    axis_vector = line_centroid - intersection
    norm = np.linalg.norm(axis_vector)
    if norm < epsilon:
        return None
    else:
        return axis_vector / norm


def find_target_pose(point_cloud, show_point_cloud=False):

    tape_points = find_points_on_tapes(point_cloud)
    if tape_points.shape[0] < 10:
        return None

    tape1_params, tape2_params = fitting_tapes_lines(tape_points)
    if tape1_params is None:
        return None
    position = estimate_intersection(tape1_params, tape2_params)

    axis_vector1 = calculate_direction(tape1_params[0], position)
    axis_vector2 = calculate_direction(tape2_params[0], position)
    if axis_vector1 is None or axis_vector2 is None:
        return None

    min_index = np.linalg.norm(tape_points[:, :3] - position,
                               axis=1).argmin()
    timestamp = tape_points[min_index, 3]

    orientation = estimate_rotation(axis_vector1, axis_vector2, position)

    if show_point_cloud:
        transformed_points = np.dot(point_cloud[:, :3] - position,
                                    orientation)
        transformed_tape_points = np.dot(tape_points[:, :3] - position,
                                         orientation)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6)
        showPointCloud([transformed_points, transformed_tape_points],
                       [coordinate])

    return position, orientation, timestamp
