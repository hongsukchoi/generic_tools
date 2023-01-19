import numpy as np
from typing import Optional, Tuple


# modified from Pytorch3d 
# referred to https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
# Returns a rotation R such that `R @ points` has a best fit plane parallel to the xy plane
def get_rotation_to_best_fit_xy(
    points: np.ndarray, centroid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    points: (N, 3) numpy matrix, points in 3D
    centroid: (1, 3) numpy matrix, their centroid

    // Return //
    rotation: (3, 3) numpy rotation matrix
    """
    if centroid is None:
        centroid = points.mean(axis=0, keepdim=True)
    
    # normalize translation
    points_centered = points - centroid

    # get covariance matrix of points, and then get the eigen vectors, 
    # which are the orthogonal each other, and thus becomes the new basis vectors
    points_covariance = points_centered.transpose(-1, -2) @ points_centered
    # eigen vectors / eigen values are in the ascending order, 
    _, evec = np.linalg.eigh(points_covariance)

    # use bigger two eigen vectors, assuming the points are flat
    rotation = np.concatenate(
        [evec[..., 1:], np.cross(evec[..., 1], evec[..., 2])[..., None]], axis=1)
    
    # `R @ points` should fit plane parallel to the xy plane
    # i.e., transform points in the eigen vector basis to the cartesian basis
    rotation = rotation.T

    # in practice, (rotation @ points.T).T
    return rotation


"""
Calculates the signed area / Lévy area of a 2D path. If the path is closed,
i.e. ends where it starts, this is the integral of the winding number over
the whole plane. If not, consider a closed path made by adding a straight
line from the end to the start; the signed area is the integral of the
winding number (also over the plane) with respect to that closed path.

If this number is positive, it indicates in some sense that the path
turns anticlockwise more than clockwise, and vice versa.
"""
# modified from Pytorch3d
# not sure what does this mean 
def _signed_area(path: np.ndarray) -> int:
    """
    path: (N, 2) numpy matrix, 2d points.

    // Returns //
    signed_area: scalar
    """
    # This calculation is a sum of areas of triangles of the form
    # (path[0], path[i], path[i+1]), where each triangle is half a
    # parallelogram.
    vector = (path[1:] - path[:1])
    x, y = vector[:, 0], vector[:, 1]
    signed_area = (y[1:] * x[:-1] - x[1:] * y[:-1]).sum() * 0.5
    return signed_area


"""
Simple best fitting of a circle to 2D points. In particular, the circle which
minimizes the sum of the squares of the squared-distances to the circle.

Finds (a,b) and r to minimize the sum of squares (over the x,y pairs) of
    r**2 - [(x-a)**2+(y-b)**2]
i.e.
    (2*a)*x + (2*b)*y + (r**2 - a**2 - b**2)*1 - (x**2 + y**2)

In addition, generates points along the circle. If angles is None (default)
then n_points around the circle equally spaced are given. These begin at the
point closest to the first input point. They continue in the direction which
seems to match the movement of points in points2d, as judged by its
signed area. If `angles` are provided, then n_points is ignored, and points
along the circle at the given angles are returned, with the starting point
and direction as before.

(Note that `generated_points` is affected by the order of the points in
points2d, but the other outputs are not.)
"""
# modified from Pytorch3d
# Returns a fitted 2D circle, which includes center, radius, and generated points


def fit_circle_in_2d(
    points2d, n_points: int = 0, angles: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   
    points2d: (N, 2) numpy matrix, 2D points
    n_points: number of points to generate on the circle, if angles not given
    angles: optional angles in radians of points to generate.

    // Returns //
    center: (2, ) numpy vector
    radius: scalar
    generated_points: (N', 2) numpy matrix, 2D points
    """

    design = np.concatenate([points2d, np.ones_like(points2d[:, :1])], axis=1)
    rhs = (points2d**2).sum(1)
    n_provided = points2d.shape[0]
    if n_provided < 3:
        raise ValueError(
            f"{n_provided} points are not enough to determine a circle")
    # solve the least sqaure problem
    solution, _, _, _ = np.linalg.lstsq(design, rhs[:, None])
    # solution: (3,1) numpy matrix

    center = solution[:2, 0] / 2  # (2*a, 2*b) / 2 -> (a, b)
    # sqrt(r**2 - a**2 - b**2 + (a**2 + b**2)) = sqrt(r*2) = r
    radius = np.sqrt(solution[2, 0] + (center**2).sum())
    if n_points > 0:
        if angles is not None:
            print("n_points ignored because angles provided")
        else:
            angles = np.linspace(0, 2 * np.pi, n_points).astype(np.float32)

    if angles is not None:
        initial_direction_xy = (points2d[0] - center)
        initial_angle = np.arctan2(
            initial_direction_xy[1], initial_direction_xy[0])

        anticlockwise = _signed_area(points2d) > 0
        if anticlockwise:
            use_angles = initial_angle + angles
        else:
            use_angles = initial_angle - angles

        generated_points = center[None, :] + radius * \
            np.stack([np.cos(use_angles), np.sin(use_angles)], axis=-1)

    else:
        generated_points = points2d

    return center, radius, generated_points


"""
Simple best fit circle to 3D points. Uses circle_2d in the least-squares best fit plane.

In addition, generates points along the circle. If angles is None (default)
then n_points around the circle equally spaced are given. These begin at the
point closest to the first input point. They continue in the direction which
seems to be match the movement of points. If angles is provided, then n_points
is ignored, and points along the circle at the given angles are returned,
with the starting point and direction as before.
""" 
# modified from Pytorch3d
# Returns a fitted 3D circle using fitted 2D circle
def fit_circle_in_3d(
    points,
    n_points: int = 0,
    angles: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    points2d: (N, 3) numpy matrix, 3D points that are assumed to be rotating around a some center
    n_points: number of points to generate on the circle
    angles: optional angles in radians of points to generate.
  
    // Returns //
    circle_points_in_3d: (N', 3) numpy matrix, evenly distribute points on the fitted circle
    normal: (3, ) numpy vector, normal of the circle plane
    """
    centroid = points.mean(0)
    projection_rotation = get_rotation_to_best_fit_xy(points, centroid)
    normal = projection_rotation.T[:, 2]

    projected_points =  (projection_rotation @ (points - centroid).T).T 
    _, _, circle_points_in_2d = fit_circle_in_2d(
        projected_points[:, :2], n_points=n_points, angles=angles
    )
    if circle_points_in_2d.shape[0] > 0:
        circle_points_in_2d_xy0 = np.concatenate(
            [
                circle_points_in_2d,
                np.zeros_like(circle_points_in_2d[:, :1]),
            ],
            axis=1,
        )
        circle_points_in_3d = (projection_rotation.T @ circle_points_in_2d_xy0.T).T + centroid
    else:
        circle_points_in_3d = points

    return circle_points_in_3d, normal

# modified from Pytorch3d
# Returns a normal that is aligned with the up vector of the world coordinate
def _disambiguate_normal(normal, up):
    """
    normal: (3,) numpy vector
    up: (3,) numpy vector
    
    // Returns //
    new_up: (3, ) numpy vector
    """
    flip = np.sign(np.sum(up * normal))
    new_up = normal * flip
    return new_up
