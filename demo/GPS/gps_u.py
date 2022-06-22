import sys
import time
from gps import *

gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)
def get_locate():
    while 1:
        report = gpsd.next()
        if report['class'] == 'TPV':
            lat = getattr(report, 'lat', 0.0)
            long = getattr(report, 'lon', 0.0)
            sys.stdout.flush() # 없어야 맞는데 안전용 print비우기
            print([1, lat, long], end="")
            sys.stdout.flush()
            time.sleep(0.0001) # 여기도 안전용
            # send image 버퍼 들어올까봐
            break