# Public-control resource successor

This preserves the completed Parakeet trace and frozen consumer, together with
both failed separate public attempts. The original 8 GiB public ceiling was
insufficient. This explicit successor declares 12 GiB RSS and requires 14 GiB
available memory before launching the same public consumer. Other guards remain
unchanged. Only metadata PermissionError writes retry for at most one second.

`run.py public` prepares and runs one new attempt; `audit.py` validates it with
the original successful trace. Both are exclusive-output commands and must not
be rerun against the closed artifact. Source differences from the original
supervisor/auditor are limited to recovery pinning and the explicit public
resource policy. There is no new numerical or runtime setting.

The [completed report](../performance-profile/results-20260921.md) records the
successful result, all three failures including the original combined worker,
and the unchanged native numerical limitations.
