from gps import *
import time
import sys

gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)
try:
    while True:
        report = gpsd.next() #
        if report['class'] == 'TPV':
            print( getattr(report,'lat',0.0),"\t")
            print( getattr(report,'lon',0.0),"\t")
            print( getattr(report,'time',''),"\t")
            print( getattr(report,'alt','nan'),"\t")
            print( getattr(report,'epv','nan'),"\t")
            print( getattr(report,'ept','nan'),"\t")
            print( getattr(report,'speed','nan'),"\t")
            print( getattr(report,'climb','nan'),"\t")
            time.sleep(1)
except KeyboardInterrupt:
    sys.exit()