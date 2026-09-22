# Correct .NET 10 instruction-width selection

The first AMD attempt stopped before cases: `DOTNET_EnableAVX512F=0` did not
select the required eight-lane product path. All owners terminated and its
failure was collected. The [.NET 10.0.8 configuration source](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/inc/clrconfigvalues.h#L615)
defines `EnableAVX512`. This successor uses `DOTNET_EnableAVX512=0` for AVX2 jobs.

First run `C:/Python313/python.exe -X utf8 -B close_failure.py`. Then run
`run.py prepare`, `run.py stage`, `run.py launch`, `run.py observe`, terminal-only
`run.py collect`, and `audit.py`. Existing destinations are refused. Product
Core `3c2f16b0` remains byte-identical. Both consumers change only their expected
environment variable literals; compiled method inspection must prove only those
two literals change in Main. All other methods and declarations remain identical.
Both complete local numerical consumers run again before payload freeze.

Exactly four fresh target processes preserve all raw/layer cases and original
numerical/ownership/resource gates. AVX512 uses the ordinary environment. Actual
product lane checks remain mandatory. The previous failed payload's fixtures
are read-only external assets, each digest-verified; only metadata is copied.
Their 509,207,936 bytes are retained in the failed staging directory. Do not
reclaim it while this successor depends on it.

There is no application timing or production selection in this qualification.
