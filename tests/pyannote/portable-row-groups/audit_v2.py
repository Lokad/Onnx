import common
common.BASE = common.ROOT / 'artifacts/pyannote-portable-row-groups-v2-20260921'
common.monitor.BASE = common.BASE
from audit import main
if __name__ == '__main__': main()
