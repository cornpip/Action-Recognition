import threading
import time
from datetime import datetime
import sys
import socket
import cv2
import numpy
import base64

# 주의! server.py 의 전역변수를 공유함
frames = []
Timing = "0"
s_msg = b""
class ServerSocket:
    def __init__(self, ip, port):
        self.TCP_IP = ip
        self.TCP_PORT = port
        self.socketOpen()
        self.receiveThread = threading.Thread(target=self.receiveImages)
        self.receiveThread.start()

    def socketClose(self):
        self.sock.close()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is close')

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.TCP_IP, self.TCP_PORT))
        self.sock.listen(1)
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is open')
        self.conn, self.addr = self.sock.accept()
        print(u'Server socket [ TCP_IP: ' + self.TCP_IP + ', TCP_PORT: ' + str(self.TCP_PORT) + ' ] is connected with client')

    def receiveImages(self):
        global frames, Timing, s_msg

        try:
            while True:
                # print("------")
                length = self.recvall(self.conn, 64)
                length1 = length.decode('utf-8')
                stringData = self.recvall(self.conn, int(length1))
                # stime = self.recvall(self.conn, 64)
                # print('send time: ' + stime.decode('utf-8'))
                # print('receive time: ' + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'))
                data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
                decimg = cv2.imdecode(data, 1)
                frames.append(decimg)
                cv2.imshow("image", decimg)

                if Timing == "1":
                    tim = Timing.encode('utf-8')
                    self.conn.send(tim)
                    self.conn.send(s_msg)
                    s_msg = b""
                else:
                    tim = Timing.encode('utf-8')
                    self.conn.send(tim)
                Timing = "0"
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            self.socketClose()
            cv2.destroyAllWindows()
            frames = [] # 연결끊을시 프레임 비우기
            self.socketOpen()
            self.receiveThread = threading.Thread(target=self.receiveImages)
            self.receiveThread.start()

    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf

def main():
    server = ServerSocket('192.168.40.21', 8080)

if __name__ == "__main__":
    main()