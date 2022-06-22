import time
from datetime import datetime
import sys
import socket
import cv2
import numpy
import base64
# from GPS import gps_u

class ClientSocket:
    def __init__(self, ip, port):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connectCount = 0
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            self.sendImages()
        except Exception as e:
            print(e)
            self.connectCount += 1
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def sendImages(self):
        FPS = 7
        cnt = 0
        prev_time = 0
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            while capture.isOpened():
                ret, frame = capture.read()
                current_time = time.time() -prev_time
                if (ret is True) and current_time > 1./FPS:
                    # resize_frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
                    prev_time = time.time()
                    stime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

                    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
                    result, imgencode = cv2.imencode('.jpg', frame, encode_param)
                    data = numpy.array(imgencode)
                    stringData = base64.b64encode(data)
                    length = str(len(stringData))
                    self.sock.sendall(length.encode('utf-8').ljust(64))
                    self.sock.send(stringData)
                    # self.sock.send(stime.encode('utf-8').ljust(64))
                    Timing = self.sock.recv(1).decode('utf-8')
                    if Timing == "1":
                        # common-0.984 charge-0.009 conflict-0.007 이 format 고정이므로
                        # buffer 크기 40으로 고정
                        inference_res = self.sock.recv(40).decode('utf-8')
                        print(inference_res)
                        inference_res = inference_res.split(' ')[:-1]
                        f_data = inference_res[0].split('-')
                        if (f_data[0] == "conflict" and float(f_data[1]) >= 0.2):
                            print("-----Dangerous-----")
                            #gps_u.get_locate()
                    if cnt % 50 == 0:
                        print('send images %d' %(cnt))
                    cnt+=1
        except Exception as e:
            print(e)
            self.sock.close()
            time.sleep(1)
            self.connectServer()
            self.sendImages()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
def main():
    TCP_IP = '118.223.255.68'
    TCP_PORT = 8001
    client = ClientSocket(TCP_IP, TCP_PORT)

if __name__ == "__main__":
    main()