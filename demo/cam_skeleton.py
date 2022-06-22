
import sys
import os
import os.path as osp
import shutil
from collections import deque
import time
import argparse
from threading import Thread
import server as socket

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

def cam_start(short_side, FPS):
    vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    prev_time = 0
    while True:
        flag, frame = vid.read()
        current_time = time.time() - prev_time
        new_h, new_w = None, None
        if (flag is True) and current_time > 1. / FPS:
            prev_time = time.time()
            # if new_h is None:
            #     h, w, _ = frame.shape
            #     new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
            #
            # frame = mmcv.imresize(frame, (new_w, new_h))
            # mmcv resize 안써도 무관해보인다.
            # 보내는 쪽 형식이 고정이 아니라면 필요할 듯
            # 처음에 frame 규격 model인자로 쓰니까

            frames.append(frame)
            if len(frames) % 10 == 0:
                print(f" |stack = {len(frames)}")

def detection_inference(args, frames):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frames))
    for frame in frames:
        result = inference_detector(model, frame)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results

def pose_inference(args, frames, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frames))
    for f, d in zip(frames, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def main():
    # global frames
    # frames = []
    # frame은 server.py 에 변수로

    args = parse_args()
    # cam = Thread(target=cam_start, args=[args.short_side, 7], daemon=True)
    #FPS설정
    # cam.start()
    # time.sleep(2)
    server = socket.ServerSocket('192.168.40.21', 8080)
    while 1:
        time.sleep(0.2)
        if len(socket.frames) > 5:
            break

    h, w, _ = socket.frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.merge_from_dict(args.cfg_options)
    for component in config.data.test.pipeline:
        if component['type'] == 'PoseNormalize':
            component['mean'] = (w // 2, h // 2, .5)
            component['max_value'] = (w, h, 1.)

    model = init_recognizer(config, args.checkpoint, args.device)
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    num_frame = 30
    alpha = int(num_frame)
    test = 0
    while True:
        if len(socket.frames) >= num_frame * 3:
            print(f" |stack-over = {len(socket.frames)}")
            if alpha <= num_frame * 3:
                print("alpha up")
                alpha += 10
            else:
                socket.frames = socket.frames[len(socket.frames)-num_frame:] # 미확인
                print("alpha set")
                alpha = int(num_frame/2)
            print(f"now alpha = {alpha}")
        if len(socket.frames) >= num_frame:
            print(f" |stack-now = {len(socket.frames)}")
            frames_c = socket.frames[:num_frame]
            socket.frames = socket.frames[num_frame+alpha:]

            det_results = detection_inference(args, frames_c)
            torch.cuda.empty_cache()

            pose_results = pose_inference(args, frames_c, det_results)
            torch.cuda.empty_cache()

            fake_anno = dict(
                frame_dir='',
                label=-1,
                img_shape=(h, w),
                original_shape=(h, w),
                start_index=0,
                modality='Pose',
                total_frames=num_frame)
            num_person = max([len(x) for x in pose_results])

            num_keypoint = 17
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

            results = inference_recognizer(model, fake_anno)
            # print("\n original result: ", results)
            s_msg = ""
            for result in results:
                act = f"{label_map[result[0]]}-{result[1]:.3f} "
                s_msg += act
            print(f"\n| {s_msg} | video num: {test}")
            if not socket.frames:
                print("| Initialize due to connection termination")
                # 1) 처리 진행 중 close했다가 연결하면 초기화 적용안됨
                # 2) 여기 조건을 지나치자 마자 연결이 끊겼다면 초기화 적용안됨
                continue
            else:
                s_msg = s_msg.encode('utf-8')
                socket.Timing = "1"
                socket.s_msg += s_msg

            action_label = label_map[results[0][0]]

            # 여기부터 저장용 영상
            pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                         args.device)
            vis_frames = [
                vis_pose_result(pose_model, frames_c[i], pose_results[i])
                for i in range(num_frame)
            ]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(args.out_filename+f"/{test}.mp4", fourcc, 18, (w, h))
            # 캠 fps를 7로 맞춰서 받고있기에 48개는 거의 7초짜리지
            for frame in vis_frames:
                cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
                out.write(frame)
            out.release()
            if test == 50:
                print(" | test=0 초기화 ")
                test = 0
            else:
                test += 1

            if cv2.waitKey(1) > 0:
                break

if __name__ == '__main__':
    main()