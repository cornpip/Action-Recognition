import argparse
import sys
import time
import os
import os.path as osp
import shutil
from collections import deque
from threading import Thread

import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from mmaction.apis import inference_recognizer, init_recognizer

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this demo! ')

try:
    from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model`, '
                      '`init_pose_model`, and `vis_pose_result` form '
                      '`mmpose.apis`. These apis are required in this demo! ')

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 255, 255)  # BGR, white
THICKNESS = 1
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default=('configs/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
        help='skeleton model config file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'),
        help='skeleton model checkpoint file/url')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument(
        '--label-map',
        default='tools/data/skeleton/label_map_ntu120.txt',
        help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def cam_start(short_side):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    flag, frame = vid.read()
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frameq_det.append(frame)
        frameq_pose.append(frame)
        flag, frame = vid.read()

def cam_detection(args):
    # model = init_detector(args["det_config"], args["det_checkpoint"], args["device"])
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    while frameq_det:
        print("=======")
        frame = frameq_det.popleft()
        result = inference_detector(model, frame)
        # args["det_score_thr"]
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        det_q.append(result)

def cam_pose(args, pose_q):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    while det_q and frameq_pose:
        print("~~~~~~~~~")
        f = frameq_pose.popleft()
        d = det_q.popleft()
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        pose_q.append(pose)
        if len(pose_q) == 30:
            print("ooooooooo")
            cam_inference(30, pose_q[::])
            pose_q = []



def cam_inference(num_frame, pose_results):
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)

    num_keypoint = 17
    num_person = max([len(x) for x in pose_results])

    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, i] = pose[:, :2]
            keypoint_score[j, i] = pose[:, 2]
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(model_recog, fake_anno)
    print("!!!!!!!!!!!!!!!!!!",results)

def main():
    global frameq_det, frameq_pose, det_q, h, w, model_recog
    frameq_det, frameq_pose = deque(), deque()
    det_q, pose_q = deque(), []

    args = parse_args()
    cam = Thread(target=cam_start, args=[args.short_side], daemon=True)
    det = Thread(target=cam_detection, args=(args,), daemon=True)
    pose = Thread(target=cam_pose, args=(args,pose_q), daemon=True)
    cam.start()
    time.sleep(1)
    det.start()
    time.sleep(1)
    pose.start()
    time.sleep(1)

    if not frameq_det[0].shape[0]:
        sys.exit("not found frame")
    h, w, _ = frameq_det[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    model_recog = init_recognizer(config, args.checkpoint, args.device)
    try:
        cam.join()
        det.join()
        pose.join()
    except KeyboardInterrupt:
        sys.exit()
    # label_map = [x.strip() for x in open(args.label_map).readlines()]

    # action_label = label_map[results[0][0]]
    #
    # pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
    #                              args.device)
    # vis_frames = [
    #     vis_pose_result(pose_model, frame_paths[i], pose_results[i])
    #     for i in range(num_frame)
    # ]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(args.out_filename, fourcc, 24, (w,h))
    # for frame in vis_frames:
    #     cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
    #                 FONTCOLOR, THICKNESS, LINETYPE)
    #     out.write(frame)
    # out.release()

if __name__ == '__main__':
    main()