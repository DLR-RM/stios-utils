"""
Visualization script.
"""

import os
import numpy as np
import cv2
from numpy import inf
from utils.utils import SENSORS, SURFACES, overlay_im_with_masks, overlay_im_with_boxes


def process_depth_for_visualization(depth):
    depth[depth < 0.] = 0.
    depth[depth > 3.] = 3.
    depth = depth / depth.max()
    return depth


def visualize(root, sensor, surface):
    assert os.path.exists(os.path.join(root, sensor, surface)), f"Invalid path: {os.path.join(root, sensor, surface)}"
    
    if sensor == 'rc_visard':
        w, h = 640, 480
    else:
        w, h = 736, 414

    for i in range(24):
        left = cv2.imread(os.path.join(root, sensor, surface, 'left_rgb', str(i).zfill(2) + '.png'))
        right = cv2.imread(os.path.join(root, sensor, surface, 'right_rgb', str(i).zfill(2) + '.png'))
        gt = cv2.imread(os.path.join(root, sensor, surface, 'gt', str(i).zfill(2) + '.png'), cv2.IMREAD_GRAYSCALE)
        depth = np.load(os.path.join(root, sensor, surface, 'depth', str(i).zfill(2) + '.npy'))
        
        if sensor == 'zed':
            # process depth
            depth = np.nan_to_num(depth)
            depth[depth == -inf] = 0
            depth[depth == inf] = 0
            depth = depth / 1000
        
        left = cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # overlay left image with ground truth annotations
        left = overlay_im_with_masks(im=left, ma=gt, alpha=0.5)
        left = overlay_im_with_boxes(im=left, ma=gt)

        # process depth for visualization
        depth = process_depth_for_visualization(depth)

        # visualize
        cv2.imshow('left + annotations', left)
        cv2.imshow('right', right)
        cv2.imshow('depth', depth)

        # load pattern depth in case of rc_visard
        if sensor == 'rc_visard':
            depth_pat = np.load(os.path.join(root, sensor, surface, 'depth_pattern', str(i).zfill(2) + '.npy'))
            depth_pat = cv2.resize(depth_pat, (w, h), interpolation=cv2.INTER_NEAREST)
            depth_pat = process_depth_for_visualization(depth_pat)
            cv2.imshow('depth+pattern', depth_pat)

        # load normals and pcd data in case of zed
        if sensor == 'zed':
            # load and process normals
            normals = np.load(os.path.join(root, sensor, surface, 'normals', str(i).zfill(2) + '.npy'))[:, :, :3]
            normals = cv2.resize(normals, (w, h), interpolation=cv2.INTER_NEAREST)
            normals = np.nan_to_num(normals)
            normals[normals == -inf] = 0
            normals[normals == inf] = 0

            # load and process point clouds
            pcd = np.load(os.path.join(root, sensor, surface, 'pcd', str(i).zfill(2) + '.npy'))[:, :, :3]
            pcd = cv2.resize(pcd, (w, h), interpolation=cv2.INTER_NEAREST)
            pcd = np.nan_to_num(pcd)
            pcd[pcd == -inf] = 0
            pcd[pcd == inf] = 0
            
            cv2.imshow('normals', normals / normals.max())
            cv2.imshow('pcd', pcd / pcd.max())
        
        cv2.waitKey(0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Visualization script for STIOS dataset")
    parser.add_argument('--root', type=str, required=True, help="Root to STIOS dataset")
    parser.add_argument('--sensor', choices=SENSORS, default=SENSORS, nargs='+', help=f"Only visualize specific sensors (default: all)")
    parser.add_argument('--surface', choices=SURFACES, default=SURFACES, nargs='+', help=f"Only visualize specific surfaces (default: all)")
    args = parser.parse_args()

    print(f"Visualizing STIOS in {args.root}")
    print(f"  sensor(s): {args.sensor}")
    print(f"  surface(s): {args.surface}")

    for sensor in args.sensor:
        for surface in args.surface:
            visualize(root=args.root, sensor=sensor, surface=surface)

