# Release benchmark consumer build correction

The original campaign stopped at compilation with two CS1061 errors:
`ITensor` exposes `Dims`; `Dimensions` belongs to typed tensors. The errors
occurred only in shape validation. No numerical or timing worker ran.
Original owner805707 / birth1790150817.56 and all children are terminal.
The complete [compiler output](initial-build-failure-20260923.txt) is retained.

Original failure closure:
`0308db404fae4312702936d57d119a04e10a921675d1a5e5fce4e1ae327f7154`.

The corrected consumer changes exactly the two property accesses to `Dims`.
Its timing code, native worker, statistical gates, output checks and resource
supervisor are byte-identical to the first version. Both sets of sources remain
under release-amd and release-amd-v2. Every fixed gate is unchanged, and the
original run has no timing samples to repeat or exclude.

The corrected campaign also retains a [preflight maintenance receipt](preflight-maintenance-20260923.json).
Between verification jobs, with no child active and before any timed worker,
the exact supervisor was paused. Content-identical files from terminal campaigns
were atomically hardlinked after full hash checks and independent local archive
verification. All918paths and SHA hashes remain;38groups reclaim737,722,708bytes.
The active namespace was excluded. Complete payload verification passed before
resuming the same supervisor. All preflight observations remain, including waits
below the unchanged12GiB launch limit. No measured interval overlapped maintenance.
