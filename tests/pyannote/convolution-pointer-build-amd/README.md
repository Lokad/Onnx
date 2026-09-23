# Normal AMD build of ordered convolution row pointers and fixed steps

Build the exact source prepared by ../convolution-row-pointer and fixed-steps using
SDK 10.0.204, the repository global.json and the existing offline NuGet feed.
Only Kernel512 may change: 3,162 other Core methods, all 697 Data methods and
public declarations must remain exact to measured Core208371f6/Data b9358370.
No new/removed methods or compiler rename exceptions are allowed. Root source
remains unchanged. The inventory checker is reused from the qualified M23 lane.

Four serial workers: SDK identity, CLI restore, ordinary Release CLI build,
then complete compiled-method inventory. CPU2 workers/CPU0 monitor; 12 GiB
available and 3 GiB tmpfs preflight, 8 GiB owned RSS, 1 GiB live available/tmpfs
floor, 900 seconds/job, 1 GiB output and 2 GiB artifacts. Preserve every
process birth identity, input hash and resource observation. Builds use
--tl:off --nologo -v minimal and disable shared compilation/build servers.

From repository root use C:/Python313/python.exe -X utf8 -B with run.py
prepare,stage,launch,observe,collect, then audit.py. Fresh artifact:
artifacts/pyannote-convolution-pointer-build-amd-20260923. Fresh VM directory:
/dev/shm/lokad-pyannote-convolution-pointer-build-20260923. Refuse existing
destinations; preserve any failed attempt. Collect only terminal owners.
Never call observe after closure or redirect audit output inside its artifact.

This lane establishes a build only. Numerical qualification must retain
original raw/wide/span/layer cases and add c64/80/128/256 coverage. Code review,
complete-call timing and all application gates follow separately.
