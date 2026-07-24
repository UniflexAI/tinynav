import cv2
import numpy as np


def reproject_depth_to_camera(src_depth, K_src, T_dst_from_src, K_dst, dst_shape, dilate_kernel_size=5):
    """Forward-splat a depth map from a source camera into a destination camera's view.

    Unprojects each valid source pixel to 3D in the source camera frame, transforms it into the
    destination camera frame via ``T_dst_from_src``, and projects it with ``K_dst``. Destination
    pixels hit by more than one source point keep the nearest (z-buffering).

    Source and destination cameras may have different resolutions/intrinsics. If the source is
    much lower resolution than the destination (e.g. stereo depth reprojected into a high-res color
    image), the raw scatter leaves most destination pixels empty; ``dilate_kernel_size`` runs a
    max-pool dilation afterward to close small gaps, trading some depth bleeding across object
    boundaries for usable coverage (acceptable here since downstream PnP uses RANSAC).
    """
    src_depth = np.asarray(src_depth, dtype=np.float32)
    K_src = np.asarray(K_src, dtype=np.float64)
    K_dst = np.asarray(K_dst, dtype=np.float64)
    T_dst_from_src = np.asarray(T_dst_from_src, dtype=np.float64)
    dst_h, dst_w = int(dst_shape[0]), int(dst_shape[1])

    fx_s, fy_s, cx_s, cy_s = K_src[0, 0], K_src[1, 1], K_src[0, 2], K_src[1, 2]
    fx_d, fy_d, cx_d, cy_d = K_dst[0, 0], K_dst[1, 1], K_dst[0, 2], K_dst[1, 2]

    v_idx, u_idx = np.nonzero(src_depth > 0.0)
    dst_depth = np.zeros((dst_h, dst_w), dtype=np.float32)
    if len(v_idx) == 0:
        return dst_depth

    z = src_depth[v_idx, u_idx].astype(np.float64)
    x = (u_idx - cx_s) * z / fx_s
    y = (v_idx - cy_s) * z / fy_s
    points_src = np.stack([x, y, z, np.ones_like(z)], axis=1)

    points_dst = (T_dst_from_src @ points_src.T).T
    z_dst = points_dst[:, 2]
    valid = z_dst > 0.0
    points_dst = points_dst[valid]
    z_dst = z_dst[valid]

    u_dst = points_dst[:, 0] * fx_d / z_dst + cx_d
    v_dst = points_dst[:, 1] * fy_d / z_dst + cy_d
    u_dst_i = np.round(u_dst).astype(np.int64)
    v_dst_i = np.round(v_dst).astype(np.int64)

    in_bounds = (u_dst_i >= 0) & (u_dst_i < dst_w) & (v_dst_i >= 0) & (v_dst_i < dst_h)
    u_dst_i = u_dst_i[in_bounds]
    v_dst_i = v_dst_i[in_bounds]
    z_dst = z_dst[in_bounds]
    if len(z_dst) == 0:
        return dst_depth

    # Nearest wins: sort farthest-first so the nearest point is scattered last per pixel.
    order = np.argsort(-z_dst)
    dst_depth[v_dst_i[order], u_dst_i[order]] = z_dst[order].astype(np.float32)

    if dilate_kernel_size and dilate_kernel_size > 1:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        dst_depth = cv2.dilate(dst_depth, kernel)

    return dst_depth
