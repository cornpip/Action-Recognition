# 운전자 폭행 감지 및 위치기반 신고 시스템  
#### 시스템에 대한 repo입니다 => [stop](https://github.com/cornpip/stop)

## dataset준비 및 모델학습
폭행 행동을 감지하기위해 mmaction2 Tool을 이용해 skeleton기반의 action-recognition인 PoseC3D를 활용합니다.

<img src="https://user-images.githubusercontent.com/74674780/178137650-09a3ab70-57f7-4f1d-8c27-f0971da147ac.jpg">  

위의 사진처럼 앞좌석 좌측에 하나 정면에 하나를 촬영 구도로 학습 데이터를 촬영합니다. 학습은 3가지 라벨로 진행합니다.  

|Common (평상시)|Conflict (폭행상황)| Charge(요금 계산 상황)|
|----|-----|----|
|351|452|71|
숫자는 학습한 영상의 수로 개당 3초 정도 길이의 영상입니다.

<img src="https://user-images.githubusercontent.com/74674780/178137612-0c77b2d0-6cd8-482c-8c20-70bc1c04895d.PNG">

학습은 위와 같은 순서로 진행됩니다.   
학습의 전처리로 각각의 영상에 객체탐지(Faster-RCNN)와 Pose-Estimation(HRNET) 작업 후 Pose의 keypoint를 pkl파일로 [__변환__](https://github.com/cornpip/mmaction2/blob/master/tools/data/skeleton/ntu_pose_extraction.py) 이 필요합니다. 각각의 영상에 대한 얻은 pkl파일을 하나의 pkl로 만들어야 합니다. => [__format_extracted__](https://github.com/cornpip/mmaction2/blob/master/tools/data/skeleton/format_extracted.py)  
해당 프로젝트에서 추출한 [__taxi_dataset.pkl__](https://github.com/cornpip/mmaction2/tree/master/data/posec3d) 입니다

사용한 config 입니다. [__taxi_keypoint__](https://github.com/cornpip/mmaction2/blob/master/configs/skeleton/posec3d/taxi_keypoint.py) 기본적으로 [__ntu60_xsub_keypoint__](https://github.com/cornpip/mmaction2/blob/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py) 환경과 동일하고 라벨 개수, gpu에러, pkl파일 위치 등의 약간의 수정이 있습니다.

준비한 pkl과(train, val 따로) config환경으로 학습을 진행합니다. => [__Train__](https://github.com/cornpip/mmaction2/blob/master/tools/train.py)  
해당 프로젝트의 학습 결과모델 입니다. => [__taxi_pth__](https://github.com/cornpip/mmaction2/tree/master/checkpoints)  

<img src="https://user-images.githubusercontent.com/74674780/178139158-5cb83f5e-c9ad-4a35-94a0-bdfa84727cfe.PNG">  
<img src="https://user-images.githubusercontent.com/74674780/178139259-b4d336b5-8c96-4b1e-8f5e-a1bb0b70c801.png">
<img src="https://user-images.githubusercontent.com/74674780/178139325-4d6532af-8ab3-4e8c-8049-53444ee5e52e.png">  
(왼쪽 상단에 추론 결과)  

## 실시간 운전자 폭행 탐지를 위한 서버

[__cam_skeleton__](https://github.com/cornpip/mmaction2/blob/master/demo/cam_skeleton.py) |  [__server_socket__](https://github.com/cornpip/mmaction2/blob/master/demo/server.py)  
cam_skeleton에 적절한 인자를 넣고 실행하여 socket연결을 대기합니다.  
_(주의: server_socket의 전역변수를 cam_skeleton에서 공유함)_
```
python demo/cam_skeleton.py [결과 영상 저장위치] `
    --config configs/skeleton/posec3d/taxi_keypoint.py `
    --checkpoint checkpoints/taxi_best_190.pth `
    --det-config demo/faster_rcnn_r50_fpn_2x_coco.py `
    --det-checkpoint http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth `
    --det-score-thr 0.9 `
    --pose-config demo/hrnet_w32_coco_256x192.py `
    --pose-checkpoint https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth `
    --label-map tools/data/skeleton/taxi_label.txt
```
추론도 앞의 학습과정과 동일하게 전처리가 필요합니다. 받은 이미지에 대해 객체를 탐지하고 pose-estimation을 한 후 keypoint값이 학습한 모델의 추론 메서드로 들어갑니다. _(그래서 --det-config, --pose-config와 같은 인자가 필요함)_

[__label.txt__](https://github.com/cornpip/mmaction2/blob/master/tools/data/skeleton/taxi_label.txt) 는 학습시킨 라벨의 순서와 맞춰 준비합니다.

__server_socket__ 은 사용하는 환경에 맞는 ip주소와 port가 필요합니다. _(server socket은 내부ip를 client socket은 외부ip를 사용함. server socket을 외부ip로 설정하면 주소를 인식하지 못하는 err가 있었다.)_  

server_socket을 연결하는 [__client_socket__](https://github.com/cornpip/stop/blob/main/pyprocess/client.py) 입니다. 해당 프로젝트에서는 node의 자식 프로세스로 실행되며 정상적으로 연결되면 정해놓은 FPS에 맞춰 이미지를 계속 전송합니다.  

__cam_skeleton__ 은 지정한 frame수에 도달하면 해당 frame수에 대해 추론을 시작하고, 결과를 client socket으로 반환하고, 결과 영상을 지정 위치에 저장합니다. _(추론이 진행 중일 때도 frame은 계속 쌓임)_  

전처리와 추론의 속도는 프로젝트에 중요한 기준입니다.  
처리 속도가 빠를수록 버려지는 frame수가 적어 정확한 감지를 할 수 있고 폭행 상황에 대한 대처도 빨라집니다. 처리 속도는 객체가 많을 수록 행동이 역동적일수록 느려집니다. 상황에 따라 처리속도가 변하기 때문에 버려지는 frame수에 대한 적절한 증가/감소가 필요합니다. 해당 증감은 서버의 __alpha__, __frames__ 변수로 다룰 수 있으므로 GPU성능과 사용 환경에 따라 적절한 값으로 수정할 수 있습니다.

## Links
#### [대한임베디드공학회 ict대학생 논문경진대회 제출 논문](https://drive.google.com/file/d/1vd5vM4-wfGYxobYWNlYLCwKDV_Oa8xU-/view?usp=sharing)  
#### [캡스톤디자인 경진대회 참여 포스터](https://docs.google.com/presentation/d/1bpxRl3pi8Qdm9mtQOApYPJharchJ7V_Y/edit?usp=sharing&ouid=109716382236660184193&rtpof=true&sd=true)   
_(로딩이 있을 수 있습니다)_

## Introduction
[mmaction2 repository](https://github.com/open-mmlab/mmaction2)

## Supported Methods

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="5" style="font-weight:bold;">Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/c3d/README.md">C3D</a> (CVPR'2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsn/README.md">TSN</a> (ECCV'2016)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README.md">I3D</a> (CVPR'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/i3d/README.md">I3D Non-Local</a> (CVPR'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/r2plus1d/README.md">R(2+1)D</a> (CVPR'2018)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/trn/README.md">TRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md">TSM</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tsm/README.md">TSM Non-Local</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowonly/README.md">SlowOnly</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/slowfast/README.md">SlowFast</a> (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/csn/README.md">CSN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tin/README.md">TIN</a> (AAAI'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tpn/README.md">TPN</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/x3d/README.md">X3D</a> (CVPR'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/omnisource/README.md">OmniSource</a> (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition_audio/resnet/README.md">MultiModality: Audio</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/tanet/README.md">TANet</a> (ArXiv'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/recognition/timesformer/README.md">TimeSformer</a> (ICML'2021)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Action Localization</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/ssn/README.md">SSN</a> (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bsn/README.md">BSN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/localization/bmn/README.md">BMN</a> (ICCV'2019)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Spatio-Temporal Action Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/acrn/README.md">ACRN</a> (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README.md">SlowOnly+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/ava/README.md">SlowFast+Fast R-CNN</a> (ICCV'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/detection/lfb/README.md">LFB</a> (CVPR'2019)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="5" style="font-weight:bold;">Skeleton-based Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/stgcn/README.md">ST-GCN</a> (AAAI'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/2s-agcn/README.md">2s-AGCN</a> (CVPR'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/README.md">PoseC3D</a> (ArXiv'2021)</td>
    <td></td>
    <td></td>
  </tr>
</table>
위와 같은 action-recognition 모델의 메서드를 사용할 수 있습니다.

## Supported Datasets

<table style="margin-left:auto;margin-right:auto;font-size:1.3vw;padding:3px 5px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="4" style="font-weight:bold;">Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hmdb51/README.md">HMDB51</a> (<a href="https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/">Homepage</a>) (ICCV'2011)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101/README.md">UCF101</a> (<a href="https://www.crcv.ucf.edu/research/data-sets/ucf101/">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/kinetics/README.md">Kinetics-[400/600/700]</a> (<a href="https://deepmind.com/research/open-source/kinetics/">Homepage</a>) (CVPR'2017)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv1/README.md">SthV1</a> (<a href="https://20bn.com/datasets/something-something/v1/">Homepage</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/sthv2/README.md">SthV2</a> (<a href="https://20bn.com/datasets/something-something/">Homepage</a>) (ICCV'2017)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/diving48/README.md">Diving48</a> (<a href="http://www.svcl.ucsd.edu/projects/resound/dataset.html">Homepage</a>) (ECCV'2018)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jester/README.md">Jester</a> (<a href="https://20bn.com/datasets/jester/v1">Homepage</a>) (ICCV'2019)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mit/README.md">Moments in Time</a> (<a href="http://moments.csail.mit.edu/">Homepage</a>) (TPAMI'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/mmit/README.md">Multi-Moments in Time</a> (<a href="http://moments.csail.mit.edu/challenge_iccv_2019.html">Homepage</a>) (ArXiv'2019)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/hvu/README.md">HVU</a> (<a href="https://github.com/holistic-video-understanding/HVU-Dataset">Homepage</a>) (ECCV'2020)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/omnisource/README.md">OmniSource</a> (<a href="https://kennymckormick.github.io/omnisource/">Homepage</a>) (ECCV'2020)</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/gym/README.md">FineGYM</a> (<a href="https://sdolivia.github.io/FineGym/">Homepage</a>) (CVPR'2020)</td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Action Localization</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/thumos14/README.md">THUMOS14</a> (<a href="https://www.crcv.ucf.edu/THUMOS14/download.html">Homepage</a>) (THUMOS Challenge 2014)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/activitynet/README.md">ActivityNet</a> (<a href="http://activity-net.org/">Homepage</a>) (CVPR'2015)</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Spatio-Temporal Action Detection</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ucf101_24/README.md">UCF101-24*</a> (<a href="http://www.thumos.info/download.html">Homepage</a>) (CRCV-IR-12-01)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/jhmdb/README.md">JHMDB*</a> (<a href="http://jhmdb.is.tue.mpg.de/">Homepage</a>) (ICCV'2015)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/ava/README.md">AVA</a> (<a href="https://research.google.com/ava/index.html">Homepage</a>) (CVPR'2018)</td>
    <td></td>
  </tr>
  <tr>
    <td colspan="4" style="font-weight:bold;">Skeleton-based Action Recognition</td>
  </tr>
  <tr>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-FineGYM</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-NTURGB+D</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-UCF101</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
    <td><a href="https://github.com/open-mmlab/mmaction2/blob/master/tools/data/skeleton/README.md">PoseC3D-HMDB51</a> (<a href="https://kennymckormick.github.io/posec3d/">Homepage</a>) (ArXiv'2021)</td>
  </tr>
</table>

위와 같은 이미 준비된 dataset을 활용할 수 있습니다.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Contributing

We appreciate all contributions to improve MMAction2. Please refer to [CONTRIBUTING.md](https://github.com/open-mmlab/mmcv/blob/master/CONTRIBUTING.md) in MMCV for more details about the contributing guideline.

## Acknowledgement

MMAction2 is an open-source project that is contributed by researchers and engineers from various colleges and companies.
We appreciate all the contributors who implement their methods or add new features and users who give valuable feedback.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their new models.

## Projects in OpenMMLab

- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
