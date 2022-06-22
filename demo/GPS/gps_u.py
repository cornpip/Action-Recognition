from gps import *

gpsd = gps(mode=WATCH_ENABLE|WATCH_NEWSTYLE)
def get_locate():
    while 1:
        report = gpsd.next()
        if report['class'] == 'TPV':
            print(getattr(report, 'lat', 0.0), "\t")
            print(getattr(report, 'lon', 0.0), "\t")
            print(getattr(report, 'time', ''), "\t")
            break