# Correction: three strided 1x1 layers were omitted from the kernel probe

The original probe selected sixteen tile geometries by excluding every1x1
convolution. That rule was wrong for this model. All three1x1layers have
stride2, while Lokad's pointwise fast path requires stride1, unit dilation,
zero padding and matching input/output spatial dimensions. These layers enter
the tiled path. The complete model has **22distinct tile geometries**, not16.

The original17.8%geometric-mean reduction and its passing controls describe
only the sixteen measured geometries. They do not establish complete kernel
coverage. The original prepared tools, measurements and closures remain intact;
this correction supersedes their claims of complete non-pointwise geometry
coverage. No sample is removed or reclassified.

| Missing rows/reduction/columns | Guarded composition selected? |
|---|---|
|64/32/472|No; short reduction retains two-row path|
|64/32/672|No; short reduction retains two-row path|
|128/64/200|Yes|
|128/64/320|Yes|
|256/128/130|Yes|
|256/128/160|Yes|

The isolated convolution implementation's108complete graph calls already
execute these layers and preserve all17.5million graph output values. Its
shared-model, full test-suite and16public-dialogue checks also pass. The running
two-meeting qualification and frozen complete-application ORT comparison retain
their original workload and gates. Their coverage is broader than the flawed
standalone shape selection and does not need to be reduced or replaced.

A separate successor must measure all22shapes under the same complete-workload
conditioning, process order and fixed controls. It must include all six omitted
shapes, including the two that retain the baseline path. Run it only after the
current application/comparison controller actually releases the local inference
lane. This is a coverage correction, not an unchanged retry or an admitted
end-to-end speedup.

The independent [metadata census](../convolution-epilogue/results-20260921.md)
pins the exact candidate source, original model and all three captured input
geometries. It independently verifies the real pointwise conditions and each
layer's stride/padding/dilation. Census closure SHA256:
`aaf2a6457d49f087b536b1966427ce1ab580554a568652f79c23a40b85ea4b84`.
