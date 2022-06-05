import cv2
import time
import os

videoFile1 = "C:\\Users\choi\PycharmProjects\\total-action\\video\\all.mp4"
# c = os.path.expanduser(videoFile1)
# print(c)
# print(videoFile1)
video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# FPS = video.get(cv2.CAP_PROP_FPS)
FPS = 24
prev_time = 0
print(FPS)

while True:

    ret, frame = video.read()

    current_time = time.time() - prev_time
    print(1/current_time)

    if (ret is True) and current_time > 1. / FPS:

        prev_time = time.time()

        print("===")

        cv2.imshow('VideoCapture', frame)

        if cv2.waitKey(1) > 0:
            break

video.release()

cv2.destroyAllWindows()