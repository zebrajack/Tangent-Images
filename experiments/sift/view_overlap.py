import torch
import torch.nn.functional as F
from mapped_convolution.transforms import *
from mapped_convolution.nn import *
from mapped_convolution.util import *

import os
import os.path as osp
import json
from skimage import io
import math
sin = math.sin
cos = math.cos


def load_rgb(root_path, basename):
    """Loads RGB"""
    img = torch.from_numpy(
        io.imread(osp.join(root_path, 'rgb',
                           basename + '_rgb.png'))[..., :3]).permute(2, 0, 1)
    mask = ~(img == 0).all(0)
    return img, mask.float()


def load_depth(root_path, basename):
    """Loads and pre-processing depth image"""
    depth = io.imread(osp.join(root_path, 'depth', basename + '_depth.png'))
    # missing values are encoded as 2^16 - 1
    missing_mask = (depth == 2**16 - 1)
    # depth 0..128m is stretched to full uint16 range (1/512 m step)
    depth = depth.astype(np.float32) / 512.0
    # clip to a pre-defined range

    depth = np.clip(depth, 0.0, 10.0)
    # zero out missing values
    depth[missing_mask] = 0.0

    return torch.from_numpy(depth)


def load_pose(root_path, basename):
    """Loads camera pose"""
    with open(osp.join(root_path, 'pose', basename + '_pose.json'), 'r') as f:
        data = json.load(f)
    return torch.tensor(data['camera_rt_matrix'])


def get_bearing_vectors(img_shape):
    # Compute bearing vectors per pixel
    lat, lon, _, _ = equirectangular_meshgrid(img_shape)
    xyz = convert_spherical_to_3d(torch.stack((lon, lat), -1))
    bearing = F.normalize(xyz, dim=-1)  # H x W x 3
    bearing = bearing.permute(2, 0, 1).contiguous()  # 3 x H x W
    return bearing


def get_in_bound_keypoints(right_img_rgb, left_img_depth, right_img_depth,
                           left_img_pose, right_img_pose, right_keypoints):
    """
    Transforms the keypoints from the right image into the left image, returning only the ones that are in bounds

    Returns K x 2 in-bounds points
    """

    # Compute bearing vectors per pixel
    bearing = get_bearing_vectors(right_img_rgb.shape[-2:])

    # Fast way to quickly get depth and bearing for keypoints only
    # Resulting shape: M, M x 3
    nn_interp_layer = Unresample('nearest')
    bisph_interp_layer = Unresample('bispherical')
    right_keypoint_depth = nn_interp_layer(
        right_img_depth.unsqueeze(0).unsqueeze(0),
        right_keypoints.unsqueeze(0)).squeeze()
    right_keypoint_bearing = bisph_interp_layer(
        bearing.unsqueeze(0), right_keypoints.unsqueeze(0)).squeeze().permute(
            1, 0)

    # Right keypoints converted to 3D points
    right_pts_3d = right_keypoint_bearing * right_keypoint_depth.view(-1, 1)
    # write_ply('right_kp.ply', right_pts_3d.permute(1, 0).numpy())

    # Transform right keypoints into left camera space
    proj_pts_3d = canonical_to_camera_transform(
        camera_to_canonical_transform(right_pts_3d, right_img_pose),
        left_img_pose)

    # Get the depth of the projected points
    proj_pts_depth = proj_pts_3d.norm(dim=-1)

    # Get the bearing vectors of the projected points
    proj_pts_bearing = F.normalize(proj_pts_3d, dim=-1)

    # Convert to 2D coordinates
    proj_in_left_img = convert_spherical_to_image(
        convert_3d_to_spherical(proj_pts_bearing), left_img_depth.shape)

    # Get the left image depth at those coordinates
    proj_in_left_depth = nn_interp_layer(
        left_img_depth.unsqueeze(0).unsqueeze(0),
        proj_in_left_img.unsqueeze(0)).squeeze()

    # Valid projected points are ones that have a lower depth than the left image depth (i.e. are visible)
    mask = proj_pts_depth < proj_in_left_depth
    proj_in_bounds = proj_in_left_img[mask]

    # Returns the K x 2 locations of the in-bounds points in the left image
    # Also returns the indicies of the right keypoints matrix that are inbounds
    return proj_in_bounds, mask.nonzero().squeeze()


# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------

data_root = '/net/vision29/data/c/2D-3D-Semantics/area_1/pano'
left_img_basename = 'camera_af500027f89e4befa60da622257156ac_WC_1_frame_equirectangular_domain'
right_img_basename = 'camera_36dd48fe958d4699ae3de7eaf4b11eb3_WC_1_frame_equirectangular_domain'
base_order = -1
sample_order = 10
scale_factor = 1
img_shape = (2048 // scale_factor, 4096 // scale_factor)


def bilinear_rescale(img, scale):
    return F.interpolate(
        img.float().unsqueeze(0),
        scale_factor=scale,
        mode='bilinear',
        align_corners=False).byte().squeeze(0)


def nn_rescale(img, scale):
    return F.interpolate(
        img.unsqueeze(0).unsqueeze(0), scale_factor=scale,
        mode='nearest').squeeze()


def camera_to_canonical_transform(pts, pose):
    """
    pts: N x 3
    pose 3 x 4
    """
    return (pts - pose[:, 3]) @ pose[:3, :3]


def canonical_to_camera_transform(pts, pose):
    """
    pts: N x 3
    pose 3 x 4
    """
    return pts @ pose[:3, :3].T + pose[:, 3]


# Load data
left_img_rgb, left_img_mask = load_rgb(data_root,
                                       left_img_basename)  # 3 x H x W, H x W
right_img_rgb, right_img_mask = load_rgb(data_root,
                                         right_img_basename)  # 3 x H x W, H x W
# When loading depth, mask out the boundaries of the image
left_img_depth = left_img_mask * load_depth(data_root,
                                            left_img_basename)  # H x W
right_img_depth = right_img_mask * load_depth(data_root,
                                              right_img_basename)  # H x W

left_img_pose = load_pose(data_root, left_img_basename)  # 3 x 4
right_img_pose = load_pose(data_root, right_img_basename)  # 3 x 4

left_img_rgb = bilinear_rescale(left_img_rgb, 1 / scale_factor)
right_img_rgb = bilinear_rescale(right_img_rgb, 1 / scale_factor)
left_img_depth = nn_rescale(left_img_depth, 1 / scale_factor)
right_img_depth = nn_rescale(right_img_depth, 1 / scale_factor)
left_img_mask = nn_rescale(left_img_mask, 1 / scale_factor)
right_img_mask = nn_rescale(right_img_mask, 1 / scale_factor)


def compute_fov_overlap_mask(left_img_depth,
                             right_img_depth,
                             left_img_pose,
                             right_img_pose,
                             eps=0.05):

    bearing = get_bearing_vectors(right_img_depth.shape)
    right_pts = right_img_depth.unsqueeze(0) * bearing
    right_pts = right_pts.view(3, -1)
    right_pts = right_pts.permute(1, 0)
    t_right_pts = canonical_to_camera_transform(
        camera_to_canonical_transform(right_pts, right_img_pose), left_img_pose)

    # Compute the number of pixels in the right image that are visible in the left image
    # Depth of the projected points
    t_right_depth = t_right_pts.norm(dim=-1)

    # Bearing vectors of the projected points
    t_right_bearing = F.normalize(t_right_pts, dim=-1)

    # Convert to 2D coordinates
    t_right_in_left_img = convert_spherical_to_image(
        convert_3d_to_spherical(t_right_bearing), left_img_depth.shape)

    # Get the left image depth at those coordinates
    nn_interp_layer = Unresample('nearest')
    t_right_in_left_depth = nn_interp_layer(
        left_img_depth.unsqueeze(0).unsqueeze(0),
        t_right_in_left_img.unsqueeze(0)).squeeze()

    # Valid projected points are ones that have a lower depth than the left image depth (i.e. are visible)
    valid_right_in_left_mask = t_right_depth <= t_right_in_left_depth + eps

    return right_pts, valid_right_in_left_mask


def compute_fov_overlap(left_img_depth,
                        right_img_depth,
                        left_img_pose,
                        right_img_pose,
                        eps=0.05,
                        save_overlap_img=False):
    right_pts, valid_right_in_left_mask = compute_fov_overlap_mask(
        left_img_depth, right_img_depth, left_img_pose, right_img_pose)
    left_pts, valid_left_in_right_mask = compute_fov_overlap_mask(
        right_img_depth, left_img_depth, right_img_pose, left_img_pose)

    right_depth_mask = right_img_depth.view(-1) >= 1e-6
    left_depth_mask = left_img_depth.view(-1) >= 1e-6
    valid_pt_mask = right_depth_mask & left_depth_mask & right_img_mask.view(
        -1).bool() & left_img_mask.view(-1).bool()
    mask = valid_right_in_left_mask & valid_left_in_right_mask & valid_pt_mask
    num_valid_right = right_pts[right_depth_mask].shape[0]
    num_valid_left = left_pts[left_depth_mask].shape[0]

    right_fov_percent = (valid_right_in_left_mask
                         & valid_pt_mask).sum().float() / num_valid_right
    left_fov_percent = (valid_left_in_right_mask
                        & valid_pt_mask).sum().float() / num_valid_right

    valid_left_img_mask = (
        valid_left_in_right_mask * valid_pt_mask).view(*left_img_depth.shape)
    valid_right_img_mask = (
        valid_right_in_left_mask * valid_pt_mask).view(*left_img_depth.shape)

    if save_overlap_img:
        left_img_overlap = left_img_rgb.clone()
        left_img_overlap[0, valid_left_img_mask] = 200
        io.imsave('left_img_overlap.png',
                  left_img_overlap.permute(1, 2, 0).numpy())
        right_img_overlap = right_img_rgb.clone()
        right_img_overlap[1, valid_right_img_mask] = 200
        io.imsave('right_img_overlap.png',
                  right_img_overlap.permute(1, 2, 0).numpy())

    return (right_fov_percent + left_fov_percent) / 2


avg_fov_overlap = compute_fov_overlap(
    left_img_depth,
    right_img_depth,
    left_img_pose,
    right_img_pose,
    save_overlap_img=True)
print('Average FOV Overlap:', avg_fov_overlap)

# left_overlap = compute_fov_overlap_mask(right_img_depth, left_img_depth,
#                                           right_img_pose, left_img_pose)
# print('Right Overlap Metric:', right_overlap)
# print('Left Overlap Metric:', left_overlap)

# # # ---------------------------------

# # Compute the keypoints on the left and right images
# # Keypoints: M x 2
# # -----------------------------
# # IF PATCH:
# resample_to_uv_layer, corners = get_tangent_plane_info(base_order, sample_order,
#                                                        left_img_rgb.shape[-2:])
# right_tex_image = resample_to_uv_layer(
#     right_img_rgb.float().unsqueeze(0)).squeeze(0).byte()

# right_keypoints = extract_sift_feats_patch(right_tex_image,
#                                            corners,
#                                            image_shape=right_img_rgb.shape[-2:],
#                                            crop_degree=30)[:, :2].contiguous()

# left_tex_image = resample_to_uv_layer(
#     left_img_rgb.float().unsqueeze(0)).squeeze(0).byte()
# left_keypoints = extract_sift_feats_patch(left_tex_image,
#                                           corners,
#                                           image_shape=left_img_rgb.shape[-2:],
#                                           crop_degree=30)[:, :2].contiguous()

# in_bound_right_keypoints, right_matches = get_in_bound_keypoints(
#     right_img_rgb, left_img_depth, right_img_depth, left_img_pose,
#     right_img_pose, right_keypoints)
# print('Patch Level {}'.format(base_order))
# print('  Number of right keypoints', right_keypoints.shape[0])
# print('  Number of right keypoints in-bounds in left image:',
#       in_bound_right_keypoints.shape[0])

# # -----------------------------
# # -----------------------------
# # IF ERP:
# right_keypoints = extract_sift_feats_erp(right_img_rgb,
#                                          crop_degree=30)[:, :2].contiguous()

# left_keypoints = extract_sift_feats_erp(left_img_rgb,
#                                         crop_degree=30)[:, :2].contiguous()

# in_bound_right_keypoints, right_matches = get_in_bound_keypoints(
#     right_img_rgb, left_img_depth, right_img_depth, left_img_pose,
#     right_img_pose, right_keypoints)
# print('Equirectangular')
# print('  Number of right keypoints', right_keypoints.shape[0])
# print('  Number of right keypoints in-bounds in left image:',
#       in_bound_right_keypoints.shape[0])

# BELOW IS ONLY USEFUL FOR PRINTING A VISUALIZATION IMAGE FOR SANITY CHECKING
# orig_keypoints = [cv2.KeyPoint(k[0], k[1], 3.0, 0) for k in right_keypoints]
# proj_keypoints = [
#     cv2.KeyPoint(k[0], k[1], 3.0, 0) for k in in_bound_right_keypoints
# ]
# left_matches = torch.arange(in_bound_right_keypoints.shape[0])
# matches = torch.stack((left_matches, right_matches), -1)
# matches = [cv2.DMatch(m[0], m[1], 0) for m in matches]

# out_img = cv2.drawMatches(left_img_rgb.permute(1, 2, 0).numpy(),
#                           proj_keypoints,
#                           right_img_rgb.permute(1, 2, 0).numpy(),
#                           orig_keypoints,
#                           matches,
#                           None,
#                           matchColor=(255, 0, 255))
# io.imsave('reproject-sift-viz.png', out_img)
