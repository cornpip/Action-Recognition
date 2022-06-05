import time

import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 열렸는지 확인
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 웹캠의 속성 값을 받아오기
# 정수 형태로 변환하기 위해 round
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)
fps = cap.get(cv2.CAP_PROP_FPS)  # 카메라에 따라 값이 정상적, 비정상적
print(fps)

# fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 1프레임과 다음 프레임 사이의 간격 설정
# delay = round(1000/fps)

# 웹캠으로 찰영한 영상을 저장하기
# cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
out = cv2.VideoWriter('output.mp4', fourcc, 25, (w, h))

print(out.isOpened())
# 제대로 열렸는지 확인
if not out.isOpened():
    print('File open failed!')
    cap.release()
    sys.exit()

while True:
    ret, img = cap.read()
    if not ret:
        print("not read")
        break

    out.write(img)
    print(type(img))
    print(img.shape)

    # time_per_frame = time.perf_counter()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
