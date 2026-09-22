# Four-block AVX512 convolution input reuse

An isolated normal-product successor to selected source `9533cd67`. Only
`ConvBlockedSpatial.Kernel512` changes: output channels divisible by 64 use a
new four-block routine. Other shapes, AVX2, layouts, weight preparation, scratch,
finite checks and ordered FMA/unfused tail arithmetic stay unchanged.

The [pinned ORT source review](../integration-review/ort-filter-blocks-20260922.md)
motivates this hypothesis. Neither source similarity nor compilation establishes
speed or installed native assembly dispatch.

From root, with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/filter-block-reuse/build.py
    tests/pyannote/filter-block-reuse/audit.py

The build archives ordinary product source into a new artifact and compares all
compiled methods against selected Core `3c2f16b0`. Exactly one existing method
may change and one private method be added; all other Core/Data methods and
public declarations must match. Root source is unchanged.

Original raw coverage uses only 32/48 output channels. A separate full raw family
adds 64/128 channels and 64-channel supplemental cases while preserving every
original family and numerical assertion. Each family has 2,648 raw cases,
20 supplemental cases, 10 invalid/alias checks, 2,668 control graph requests and
5,336 prepared graph requests. All 108 captured layer graphs also run. Local AVX2
results do **not** qualify the new AVX512 kernel; both AMD widths and separate
generated-code inspection precede any complete-call performance comparison.

Freeze each input before execution, retain all clocks/resources and failures,
refuse existing destinations and audit only terminal owners. No performance
measurement, root integration or parity claim is made by this local protocol.
