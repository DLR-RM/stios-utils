"""
Various utility definitions and functions.
"""

import os
import numpy as np
import cv2
from ruamel import yaml

from utils.colormap import get_spaced_colors


SENSORS = [
    'rc_visard',
    'zed'
]

SURFACES = [
    'conveyor_belt',
    'lab_floor',
    'office_carpet',
    'robot_workbench',
    'tool_cabinet',
    'white_table',
    'wooden_plank',
    'wooden_table'
]

YCB_OBJECTS = [
    '003_cracker_box',
    '005_tomato_soup_can',
    '006_mustard_bottle',
    '007_tuna_fish_can',
    '008_pudding_box',
    '010_potted_meat_can',
    '011_banana',
    '019_pitcher_base',
    '021_bleach_cleanser',
    '024_bowl',
    '025_mug',
    '035_power_drill',
    '037_scissors',
    '052_extra_large_clamp',
    '061_foam_brick',
]


def overlay_im_with_masks(im, ma, alpha=0.5):
    """
    Overlays an image with corresponding annotations.
    :param im: (uint8 np.array) image of shape h, w, 3
    :param ma: (uint8 np array) mask of shape h, w; expects unique integers for object instances
    :param alpha: (float) see cv2.addWeighted() for more information
    :return: (uint8 np.array) colorized image of shape h, w, 3
    """

    if ma.max() == 0:
        return im
    colors = get_spaced_colors(20)
    im_col = im.copy()
    for ctr, i in enumerate(np.unique(ma)[1:]):
        a, b = np.where(ma == i)
        if a != []:
            im_col [a, b, :] = colors[ctr]
    im_overlay = im.copy()
    im_overlay = cv2.addWeighted(im_overlay, alpha, im_col, 1 - alpha, 0.0)
    return im_overlay


def overlay_im_with_boxes(im, ma):
    """
    Overlays an image with corresponding bounding box annotations.
    :param im: (uint8 np.array) image of shape h, w, 3
    :param ma: (uint8 np array) mask of shape h, w; expects unique integers for object instances
    :return:
    """

    im_boxann = im.copy()
    dets = get_bboxes(ma)
    for det in dets:
        cl = det['class']
        bbox = det['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        im_boxann = cv2.rectangle(im_boxann, (x, y), (x + w, y + h), (36, 255, 12), 1)
        cv2.putText(im_boxann, cl, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    return im_boxann


def get_bboxes(ma):
    """
    Creates a list of dicts of bounding box annotations with corresponding class label for each unique instance.
    :param ma: (uint8 np array) mask of shape h, w; expects unique integers for object instances
    :return: (list) list of dicts with bounding box annotations
    """

    dets = []
    for un in np.unique(ma)[1:]:
        x, y, w, h = cv2.boundingRect(((ma == un) * 1).astype(np.uint8))
        dets.append({
            'class': YCB_OBJECTS[un - 1],
            'bbox': {
                'x': x,
                'y': y,
                'w': w,
                'h': h
            }
        })

    return dets


def get_params(sensor):
    assert sensor in SENSORS, f"Invalid sensor: {sensor}. Supported: {SENSORS}"

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../params', sensor + '.yaml'), 'r') as f:
        params = yaml.load(f)

    return params


def depth_to_disparity(depth, sensor):
    assert sensor in SENSORS, f"Invalid sensor: {sensor}. Supported: {SENSORS}"

    params = get_params(sensor=sensor)

    h, w = params['depth']['height'], params['depth']['width']
    assert depth.shape == (h, w), f"Depth shape changed from ({h}, {w}) to {depth.shape}"

    fx = params['depth']['fx']
    baseline = params['color']['baseline']

    scaling_factor = 1. if params['depth']['unit'] == 'm' else 0.001
    depth = depth * scaling_factor

    disp = np.zeros_like(depth)
    disp[depth != 0] = fx * baseline / depth[depth != 0]

    return disp
