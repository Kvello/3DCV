""" CS4277/CS5477 Lab 1: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab1.pdf) for instructions.

Name: Markus Kvello
Email: e1163359@u.nus.edu
Student ID: A0283551A
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt
import part1 as p1
import part2 as p2


def warp_image_on_canvas(image, h_matrix):
    """Warps the image onto a black canvas using the provided homography matrix
     in such a way that the entire warped image is visible.

    Args:
        image (np.ndarray): Image to warp
        h_matrix (np.ndarray): Homography matrix that transforms image to the
          canvas, i.e. x_{canvas} = h_matrix * x_{image},
        where x_{image}, x_{canvas} are the homogeneous coordinates in I_{image}
        and I_{canvas} respectively
    Returns:
        canvas (np.ndarray): Warped image on canvas

     """

    corners_transformed = []
    h, w = image.shape[:2]
    bounds = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    transformed_bounds = transform_homography(bounds, h_matrix)
    corners_transformed.append(transformed_bounds)
    corners_transformed = np.concatenate(corners_transformed, axis=0)
    corners_transformed = np.array(corners_transformed)
    # Compute required canvas size
    min_x, min_y = np.min(corners_transformed, axis=0)
    max_x, max_y = np.max(corners_transformed, axis=0)
    min_x, min_y = floor(min_x), floor(min_y)
    max_x, max_y = ceil(max_x), ceil(max_y)

    canvas = np.zeros((max_y-min_y, max_x-min_x, 3), image.dtype)

    # adjust homography matrix
    trans_mat = np.array([[1.0, 0.0, -min_x],
                          [0.0, 1.0, -min_y],
                          [0.0, 0.0, 1.0]], h_matrix.dtype)
    h_adjusted = trans_mat @ h_matrix

    # Warp
    canvas = warp_image(image, canvas, h_adjusted)

    return canvas

def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr*2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    src = src.copy()
    src = np.vstack((src.T, np.ones((1, src.shape[0]))))
    h_matrix = np.matrix(h_matrix)
    transformed = h_matrix * src
    transformed = transformed / transformed[2, :]
    transformed = transformed[:2, :].T
    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    h_matrix = np.matrix(h_matrix)
    dst = dst.copy()  # deep copy to avoid overwriting the original image
    x, y = np.meshgrid(np.arange(dst.shape[1]), np.arange(dst.shape[0]))
    M = np.vstack((x.flatten(), y.flatten()))
    M_transformed = transform_homography(M.T, np.linalg.inv(h_matrix)).T
    x_map = M_transformed[0].reshape(
        dst.shape[0], dst.shape[1]).astype(np.float32)
    y_map = M_transformed[1].reshape(
        dst.shape[0], dst.shape[1]).astype(np.float32)
#    M_transformed= np.clip(M_transformed, [[0],[0]],[[dst.shape[1] - 1],[dst.shape[0] - 1]])
    cv2.remap(src, x_map, y_map, interpolation=cv2.INTER_LINEAR, dst=dst,
              borderMode=cv2.BORDER_TRANSPARENT)
    # Without using cv2.remap
    # for (x,y) in M.T:
    #     if(x_map[y,x] >= 0 and x_map[y,x] < src.shape[1] and y_map[y,x] >= 0 and y_map[y,x] < src.shape[0]):
    #         dst[y,x] = src[int(y_map[y,x]),int(x_map[y,x])]

    # cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
    #                     dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst


def compute_affine_rectification(src_img: np.ndarray, lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Affinely rectified image by removing projective distortion

    '''
    Hp = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    assert (len(lines_vec) >= 2 and len(lines_vec) % 2 == 0)
    Q = np.zeros((3, 3))
    crossing_points = []
    for l1, l2 in zip(lines_vec[::2], lines_vec[1::2]):
        # Find the intersection of the two lines
        l1 = l1.vec_para
        l2 = l2.vec_para
        X = np.matrix(np.cross(l1, l2)).T
        X = X / X[-1]
        crossing_points.append(X)
        # given by the cross product of the two lines(homogeneous coordinates)
        # Check that the intersection is not at infinity ???
        # Update Q (see report for details)
        Q += X*X.T
    # Now we solve for the null space of Q, which gives us the least square solution for the vanishing line
    # See report for details
    U, S, Vt = np.linalg.svd(Q)
    l_inf = np.matrix(Vt[-1]).T

    # Now we can construct the projective transformation matrix Hp
    Hp = np.eye(3)
    Hp[2] = l_inf.T
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    # cv2.warpPerspective(src_img, Hp, (src_img.shape[1], src_img.shape[0]), dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    dst = warp_image_on_canvas(src_img, Hp)
    """ YOUR CODE ENDS HERE """


    ############## Debugging ################


#    # Draw the vanishing points and the line on the image
#    dst_debug = src_img.copy()
#    # Zoom out picture to see the vanishing points and line
#
#    height, width = dst_debug.shape[:2]
#    
#    # Define the four corners of the original image
#    pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
#    
#    # Define the four corners of the desired warped image
#    center_x = width / 2
#    center_y = height / 2
#    offset = 100 # You can adjust this value for more or less squishing
#    pts2 = np.float32([[center_x - offset, center_y - offset],
#                       [center_x + offset, center_y - offset],
#                       [center_x + offset, center_y + offset],
#                       [center_x - offset, center_y + offset]])
#    
#    # Compute the perspective transformation matrix
#    matrix = cv2.getPerspectiveTransform(pts1, pts2)
#    
#    # Apply the perspective transformation to the image
#    dst_debug = cv2.warpPerspective(dst_debug, matrix, (width, height))
#    crossing_points = np.array(list(map(lambda X: matrix @ X, crossing_points)))
#    for X in crossing_points:
#        cv2.circle(dst_debug, (int(X[0]), int(X[1])), 5, (0, 0, 255), -1)
#    # Draw the vanishing line
#    x = np.arange(0, width, 1)
#    l_inf_dbg = np.linalg.inv(matrix).T @ l_inf
#    y = (-l_inf_dbg[0, 0] * x - l_inf_dbg[2, 0]) / l_inf_dbg[1, 0]
#    y = y.astype(int)
#    for i in range(len(x)):
#        cv2.circle(dst_debug, (x[i], y[i]), 2, (0, 255, 0), 5)
#    cv2.polylines(dst_debug, [np.column_stack((x, y))], False, (0, 255, 0), 1)
#    
#    # Display the result
#    cv2.imshow('Warped Image', dst_debug)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    return dst


def compute_metric_rectification_step2(src_img: np.ndarray, line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           X_dst: Image after metric rectification

    '''
    dst = np.zeros_like(
        src_img)  # deep copy to avoid overwriting the original image
    Ha = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    assert(len(line_vecs) >= 2 and len(line_vecs) % 2 == 0)
    Q = np.zeros((len(line_vecs)//2, 3))
    for i,lines in enumerate(zip(line_vecs[::2], line_vecs[1::2])):
        m, l = lines
        m_vec = m.vec_para
        l_vec = l.vec_para
        Q[i] = np.array([m_vec[0]*l_vec[0], m_vec[0]*l_vec[1] + m_vec[1]*l_vec[0], m_vec[1]*l_vec[1]])
    U, sigma, Vt = np.linalg.svd(Q)
    null_vec = Vt[-1]
    S = np.matrix([[null_vec[0], null_vec[1]], [null_vec[1], null_vec[2]]])
    S = 1/abs(np.linalg.det(S))**(1/2) *S
    K = np.linalg.cholesky(S)
    Ha = np.block([[K, np.zeros((2,1))], [np.zeros((1,2)), 1]])
    Ha = np.linalg.inv(Ha)
    dst = warp_image_on_canvas(src_img,Ha)
    cv2.imshow('Warped Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """ YOUR CODE ENDS HERE """

    return dst


def compute_metric_rectification_one_step(src_img: np.ndarray, line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           line_infinity: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Image after metric rectification

    '''
    dst = np.zeros_like(
        src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """
    assert(len(line_vecs) >= 2 and len(line_vecs) % 2 == 0)
    Q = np.zeros((len(line_vecs)//2, 6))
    for i,lines in enumerate(zip(line_vecs[::2], line_vecs[1::2])):
        m_line, l_line = lines
        m = m_line.vec_para
        l = l_line.vec_para
        Q[i] = np.array([l[0]*m[0], 
                         (l[0]*m[1] + l[1]*m[0])/2, 
                         l[1]*m[1], 
                         (l[0]*m[2]+l[2]*m[0])/2,
                         (l[1]*m[2]+l[2]*m[1])/2, 
                         l[2]*m[2]])
        
    U, sigma, Vt = np.linalg.svd(Q)
    null_vec = Vt[-1]
    C_inf_img = np.matrix([[null_vec[0], null_vec[1]/2, null_vec[3]/2],
                           [null_vec[1]/2, null_vec[2], null_vec[4]/2],
                           [null_vec[3]/2, null_vec[4]/2, null_vec[5]]])

    U, sigma, Vt = np.linalg.svd(C_inf_img)
    H = U.T
    s = 0.42
    theta = np.pi*1.2
    tx = 20
    ty = 100
    Hs = np.matrix([[s*np.cos(theta)*s,-np.sin(theta)*s,tx],
                    [s*np.sin(theta),s*np.cos(theta),ty],
                    [0,0,1]])
    H = Hs*H # Apply similarity transformation to make image visible 
    H_inv = np.linalg.inv(H)
#     for x,y in zip(range(src_img.shape[1]), range(src_img.shape[0])):
#         X = np.array([x,y,1])
#         X = X.reshape(3,1)
#         X_img= H_inv @ X
#         X_img = X_img / X_img[2]
#         print("The point (",x,y,") gets mapped to, : ", X_img.T[0,0:2])
    dst = warp_image_on_canvas(src_img,H)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)
    dst = np.block([dst, np.ones((dst.shape[0], 1))])
    src = np.block([src, np.ones((src.shape[0], 1))])
    """ YOUR CODE STARTS HERE """
    X = np.linalg.inv(homography)@dst.T
    X = X / X[2]
    X_img = homography@src.T
    X_img = X_img / X_img[2]
    d =  np.linalg.norm(src.T-X,ord=2,axis=0)**2 +\
            np.linalg.norm(dst.T-X_img,ord=2,axis=0)**2
    """ YOUR CODE ENDS HERE """

    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """
    mask = np.ones(src.shape[0], dtype=bool)
    h_matrix = np.zeros(src.shape[0], dtype=bool)
    biggest_inlier_mask = np.zeros(src.shape[0], dtype=bool)
    """ YOUR CODE STARTS HERE """
    for _ in range(num_tries):
        # Randomly sample 4 correspondences
        idx = np.random.choice(src.shape[0], 4, replace=False)
        src_sample = src[idx]
        dst_sample = dst[idx]
        # Compute homography
        h_model = compute_homography(src_sample, dst_sample)
        # Compute error
        d = compute_homography_error(src, dst, h_model)
        # Update mask
        mask = d < thresh
        # Update homography
        if np.sum(mask) > np.sum(biggest_inlier_mask):
            biggest_inlier_mask = mask
    mask = biggest_inlier_mask
    h_matrix = compute_homography(src[mask], dst[mask])
    """ YOUR CODE ENDS HERE """

    return h_matrix, mask


