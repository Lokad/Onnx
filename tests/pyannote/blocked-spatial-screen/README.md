# Build the fixed component benchmark

Copy the exact qualified `ModelProbe`, `BlockedSpatial` and generated kernels
from their closed artifacts. Add a separate benchmark entry point. Its
`qualify` mode invokes the unchanged model probe and must preserve every
original observation before the executable can be transferred to AMD.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-screen/build.py
    tests/pyannote/blocked-spatial-screen/audit_build.py

The Windows qualification is closed at
`1412473e7eae6ae69f42c2f47bdf589b8b765014cc34e96f6167ee1fc96fda32`.
All 108 cases and 119,823,360 values pass exactly; all 53 resource samples pass.
No local performance measurement is taken. The fixed target campaign is in
[blocked-spatial-screen-amd](../blocked-spatial-screen-amd/README.md).
