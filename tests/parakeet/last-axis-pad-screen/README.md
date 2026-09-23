# Complete public Pad comparison

Prospective protocol, before any measured calls. The selected release is
94a550de (Core 521bae17); the distinct padding build is Core 5cf89ae3, closure
62043d10. Six public-operator tests pass in each of two instruction modes, and the exact
PadCore composition review passes. All matrix methods and flags remain selected.

Four fresh processes run selected, candidate, candidate, selected on AMD CPU 2.
Each case performs 600 fixed warmups and 180 measurements. Keep all 37,440 calls,
28,800 warmups, 8,640 measurements, 48 setups and 144 fixed measured blocks.
The timing boundary is the complete public CPUExecutionProvider.Pad call:
validation, materialization, allocation, fill, mapping/copy and returned owned
tensor. Fixture setup, oracle, output verification and ownership mutations are
outside it. No profiler, forced collection or runtime override is permitted.

`census.py` fixes all shapes before execution. Eight primary cases combine
recorded frame counts 51/106/167/225 with synthetic attention shapes [1,8,T,2T-1]
and convolution shapes [1,1024,T]. These are constructed examples, not captured
Pad intermediates. Their respective last-axis pads are (1,0) and (4,4), matching
all 48 actual encoder Pad nodes. Four coverage cases exercise zero padding,
cropping, outer-axis padding and reflection. The last three keep the old path.

All outputs must match an independent destination-coordinate oracle bit for
bit; selected/candidate hashes must agree. Shapes, unchanged inputs and held
output independence are checked. The same common consumer serves both products.
Use all measured clocks and exact rational arithmetic with equal process weights.
Require 26 same-engine controls (each case and the eight-case sum) <=1.10,
all twelve candidate/selected means <=1.05, the eight-case sum at least 20%
faster, and strict separation between both candidate and both selected sums.
Retain every failure; no trimming, favorable block selection or unchanged retry.

Preflight 12 GiB available/3 GiB tmpfs, worker RSS <8 GiB, each job <900 seconds,
at least 1 GiB available/tmpfs throughout, <=1 GiB per-job output, <=2 GiB campaign
files and four hours total. CPU 0 monitors. SDK 10.0.204, .NET 10.0.8, normal
build under source/global.json with --tl:off. No model copies or root changes.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with
`run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`.
Source/build prerequisites and the composition proof are mandatory. An admitted
component screen permits complete Parakeet qualification; it does not select
a release or update BENCHMARK.md.
