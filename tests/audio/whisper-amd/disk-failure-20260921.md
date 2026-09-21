# AMD Whisper comparison stopped by transient disk exhaustion

The fixed comparison is **incomplete**. Conformance completes20/20 calls, followed
by three complete80-call timing workers: managed, ORT and ORT. The final managed
worker stops after34/80 calls when free disk falls to **598,016 bytes**, below the
unchanged33,554,432-byte reserve. Available memory is **5,086,019,584 bytes**;
the other resource guards still pass. This is a disk failure, not a memory failure
or observed application mismatch.

All260 calls from complete workers pass original request, identity, ownership
and resource checks. The34saved partial calls also match the original application
results and their per-request input/ownership checks. They do not establish that
the interrupted request or final held-output check completed. Every prior sample
in the failed worker passes; its2064th sample triggers the stop. All six original
process identities are terminal. No worker is restarted or partial comparison scored.

The final disk samples move from82,522,112 to67,248,128, recover to82,735,104,
then fall to598,016bytes. The journal records apt-daily starting at03:40:53UTC and
PackageKit at03:40:54, when the guard fires. Two roughly61MB package-cache files
were rewritten at03:40:57; free disk subsequently recovered to about83MB. These
observations support attributing the transient exhaustion to package-cache work;
they are not a byte-by-byte allocation trace.

Under the exclusive-VM authorization, apt-daily, apt-daily-upgrade, their timers
and PackageKit are temporarily stopped and masked in `/run`. The saved action
receipt retains prior states and exact restoration commands. Restore those units
after the benchmark window. No model or unique evidence was removed.

The matched AMD timing result remains pending; the Windows Whisper baseline and
completed AMD Parakeet/pyannote baselines retain their original scope. A storage
correction must precede a separately declared replacement comparison. Successful
conformance can be reused after identity verification; this failed timing campaign
must remain visible. e5's independent prepared campaign may proceed after verifying
these terminal identities and closure; its correctness and A/A gates are unchanged.

Frozen SHA256: `0fafe2763d111a63a4bc0f8f5aad45eb3a911e2ade23d9dacfde0eeab6a7e74d`. Collection SHA256:
`0d11de0b6cdda3c443120ab371f29b9ee8cc8c00da77d7dc990b5701e64fe166`. Artifact:
`artifacts/audio-whisper-amd-20260921`. Every raw record, sample, traceback,
collection receipt, journal observation and service-action receipt is retained.
