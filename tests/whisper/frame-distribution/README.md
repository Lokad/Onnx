# Position of the saved Whisper encoder discrepancies

`analyze.py --artifact <fresh-directory>` analyzes all 84 outputs from the closed
same-input comparison. It loads no inference engine and changes no acceptance
criterion. Every reused array, recording length and graph is checked against
the retained receipts.

The tool stores all per-frame counts, maximum scaled differences and squared
differences for all six original terms. It checks these against the previous
complete-array audit and verifies exact repeated-case metrics. Reports separate
the unique twenty clips from twenty-one requests including the repeat.

Regions use the local input footprint before attention: encoder position `j`
has center `320*j` samples and a conservative footprint extending 520 samples
on either side, from the centered Fourier window and both convolution kernels.
Rows lie before the PCM end, cross the end, or lie after it. Attention and the
frontend's clipping floor depend on the entire clip. Consequently a position
after the PCM end is still a real output whose numerical check must pass.

This is descriptive analysis selected after the full experiment, not a new
prospective acceptance test. The original full-array failures remain failures.
