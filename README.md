# 운전자 폭행 감지 및 위치기반 신고 시스템  
## 구조
<img src="https://user-images.githubusercontent.com/74674780/178137022-1b64682c-4ed6-48d6-9c39-05b08b2503c0.PNG" width=700>

시스템의 흐름은 다음과 같습니다. 추론에는 gpu가 필요하므로 추론 서버를 분리합니다. Raspberry Pi에서 [__추론 서버에 socket 연결__](https://github.com/cornpip/stop/blob/main/pyprocess/client.py) 을 한 후 이미지를 전송합니다. 서버는 들어온 이미지를 정해진 프레임 단위로 추론을 진행하고 결과를 반환합니다. 결과가 폭행 상황일시 Pi와 연결된 GPS모듈로 부터 [__현재 위도/경도__](https://github.com/cornpip/stop/blob/main/pyprocess/gps_u.py) 를 받아오고 부모 프로세서인 [__node process__](https://github.com/cornpip/stop/blob/main/index.js) 에 전달합니다. node procecss는 준비된 csv파일로 부터 경찰서의 위도/경도를 비교하고 [__가장 가까운 경찰서를 반환__](https://github.com/cornpip/stop/blob/main/track/shortcut.js)합니다.  

<img src="https://user-images.githubusercontent.com/74674780/178142558-f2243126-3d77-4532-a851-a83ddc6ca5b1.jpg" width=700> 

네이버의 Simple & Easy Notification Service를 이용합니다. 사용 가이드에 따라 Signature를 생성하고 api를 호출합니다.    

## dataset준비 및 모델학습
폭행 행동을 감지하기위해 mmaction2 Tool을 이용해 skeleton기반의 action-recognition인 PoseC3D를 활용합니다.

<img src="https://user-images.githubusercontent.com/74674780/178137650-09a3ab70-57f7-4f1d-8c27-f0971da147ac.jpg" width=600>

위의 사진처럼 앞좌석 좌측에 하나 정면에 하나를 촬영 구도로 학습 데이터를 촬영합니다. 학습은 3가지 라벨로 진행합니다.  

|Common (평상시)|Conflict (폭행상황)| Charge(요금 계산 상황)|
|----|-----|----|
|351|452|71|
숫자는 학습한 영상의 수로 개당 3초 정도 길이의 영상입니다.

<img src="https://user-images.githubusercontent.com/74674780/178137612-0c77b2d0-6cd8-482c-8c20-70bc1c04895d.PNG" width=700>

학습은 위와 같은 순서로 진행됩니다.   
학습의 전처리로 각각의 영상에 객체탐지(Faster-RCNN)와 Pose-Estimation(HRNET) 작업 후 Pose의 keypoint를 pkl파일로 [__변환__](https://github.com/cornpip/mmaction2/blob/master/tools/data/skeleton/ntu_pose_extraction.py) 이 필요합니다. 각각의 영상에 대한 얻은 pkl파일을 하나의 pkl로 만들어야 합니다. => [__format_extracted__](https://github.com/cornpip/mmaction2/blob/master/tools/data/skeleton/format_extracted.py)  
해당 프로젝트에서 추출한 [__taxi_dataset.pkl__](https://github.com/cornpip/mmaction2/tree/master/data/posec3d) 입니다

사용한 config 입니다. [__taxi_keypoint__](https://github.com/cornpip/mmaction2/blob/master/configs/skeleton/posec3d/taxi_keypoint.py) 기본적으로 [__ntu60_xsub_keypoint__](https://github.com/cornpip/mmaction2/blob/master/configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py) 환경과 동일하고 라벨 개수, gpu에러, pkl파일 위치 등의 약간의 수정이 있습니다.

준비한 pkl과(train, val 따로) config환경으로 학습을 진행합니다. => [__Train__](https://github.com/cornpip/mmaction2/blob/master/tools/train.py)  
해당 프로젝트의 학습 결과모델 입니다. => [__taxi_pth__](https://github.com/cornpip/mmaction2/tree/master/checkpoints)  

<img src="https://user-images.githubusercontent.com/74674780/178139158-5cb83f5e-c9ad-4a35-94a0-bdfa84727cfe.PNG">  
<img src="https://user-images.githubusercontent.com/74674780/178139259-b4d336b5-8c96-4b1e-8f5e-a1bb0b70c801.png">
<img src="https://user-images.githubusercontent.com/74674780/178139325-4d6532af-8ab3-4e8c-8049-53444ee5e52e.png">  
<img src="https://github.com/cornpip/Action-Recognition/assets/74674780/057db13c-f352-4251-9fea-76e578099e27" width=600 />

왼쪽 상단에 추론 결과를 확인할 수 있습니다.

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

## 라즈베리파이 구성
<img src="https://github.com/cornpip/Action-Recognition/assets/74674780/b2e385fb-0ea5-4db3-a594-9c396e1b98aa" width=400 />


## Links
#### [대한임베디드공학회 ict대학생 논문경진대회 제출 논문](https://drive.google.com/file/d/1vd5vM4-wfGYxobYWNlYLCwKDV_Oa8xU-/view?usp=sharing)  
#### [캡스톤디자인 경진대회 참여 포스터](https://docs.google.com/presentation/d/1bpxRl3pi8Qdm9mtQOApYPJharchJ7V_Y/edit?usp=sharing&ouid=109716382236660184193&rtpof=true&sd=true)  
#### [운전자 폭행 감지 라즈베리파이 서버 Repository](https://github.com/cornpip/stop)
#### [mmaction2 Repository](https://github.com/open-mmlab/mmaction2)
