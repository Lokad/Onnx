# Wide projection guard extension build

Normal AMD build of M52, with the only semantic source delta from M50 removing
the isolated dispatcher's `m < 64` guard. All original axis/budget/ISA guards,
arithmetic, method names, flags and public APIs remain. The selected root is
unchanged. The historical private ShortWide names are retained for exact scope.

The complete compiled inventory compares actual M50 and new products: 3185 Core
methods with 3184 exact, one exact guard removal, 697 Data methods exact, equal
flags and public APIs, no method additions/removals or compiler rename. The
inspector is reused unchanged. Tests reject unauthorized body, branch, flag,
fallback and census changes. No numerical or performance admission follows a build.

Run `C:/Python313/python.exe -X utf8 -B -m unittest discover -s
tests/parakeet/wide-projection-isolation-build-amd -p test_checks.py`, then run.py
prepare, stage, launch, observe, collect separately and audit.py after termination.
Freeze and commit before VM execution. Exclusive namespaces refuse reuse.

Four serial jobs: SDK, offline restore, normal Release build and full inventory.
CPU2 build/inventory, CPU0 monitoring; every thread checked. Bounds are 10GiB
available/3GiB tmpfs preflight, 8GiB RSS, 1GiB available/tmpfs minimum, 1GiB
output/job, 2GiB campaign, 900seconds/job, 4hours overall, monitoring gaps<10seconds.
