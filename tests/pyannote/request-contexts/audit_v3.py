"""Apply the same application qualification audit to the explicit-overload successor."""
from common import *
import audit

if __name__ == '__main__':
    target = ROOT / 'artifacts/pyannote-request-contexts-v3-20260921'
    assert pin(target / 'prepared.json')['sha256'] == '7f09769784068c7398ac72aae02d39c92647aafcbedae1c1775221caeeb6a50d'
    audit.BASE = target
    audit.main()
