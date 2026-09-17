# Proof for AMD-01 process accounting (M01 red-proof pattern).
# Unit legs run anywhere; -Live runs controlled foreign-workload detection on
# an idle box OUTSIDE any campaign (it burns one CPU for ~12s by design).
param([switch]$Live)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "proc-acct.ps1")
$fail = 0
function Check($cond, $name) {
  if ($cond) { Write-Host ("ok - " + $name) }
  else { Write-Host ("FAIL - " + $name); $script:fail++ }
}
# UNIT 1: real ps format (bare integer seconds, padded columns, etime ident).
$script:ProcParseErrors = 0
$rows = Parse-ProcLines @(
  "      1       0        5 02:11:33 systemd",
  "   9136    9120     1200 00:00:02 pwsh",
  "     42       2        0 1-00:00:00 kworker/x",
  "bad line",
  "12 34",
  "  99    98 abc0932 00:00:01 myproc")
Check ($rows.Count -eq 3) "three good rows parsed"
Check (($rows | Where-Object { $_.Id -eq 1 }).CPUSec -eq 5.0) "integer seconds read as 5"
Check (($rows | Where-Object { $_.Id -eq 9136 }).CPUSec -eq 1200.0) "large integer seconds read"
Check (($rows | Where-Object { $_.Id -eq 1 }).Elapsed -eq "02:11:33") "etime kept as identity"
Check ($script:ProcParseErrors -eq 3) "three malformed rows counted, none zero-filled"
# UNIT 2: ownership walk (cycle-safe, PID-rooted).
$map = @{ $PID = @(0, 0); 4242 = @(0, $PID); 9999 = @(0, 4242); 7777 = @(0, 7777) }
Check ((Test-LaneOwned 9999 $map) -eq $true) "descendant detected"
Check ((Test-LaneOwned 4242 $map) -eq $true) "child detected"
Check ((Test-LaneOwned 123456 $map) -eq $false) "stranger declined"
Check ((Test-LaneOwned 7777 $map) -eq $false) "self-cycle terminates declined"
if ($Live) {
  if (-not $IsLinux) { Write-Host "SKIP - live legs are Linux-only"; exit ($fail -gt 0 ? 1 : 0) }
  function Frac10 {
    $a = @{}; foreach ($r in (Get-ProcTable)) { $a[$r.Id] = @($r.CPUSec, $r.PPid) }
    Start-Sleep -Seconds 10
    $f = 0.0
    foreach ($r in (Get-ProcTable)) {
      if ($r.Id -eq 0 -or (Test-LaneOwned $r.Id $a)) { continue }
      if ($a.ContainsKey($r.Id)) { $d = $r.CPUSec - $a[$r.Id][0]; if ($d -gt 0) { $f += $d } }
    }
    return $f / 40.0
  }
  $idle = Frac10
  Write-Host ("idle-10s frac=" + $idle.ToString("F3"))
  Check ($idle -lt 0.02) "idle box reads idle"
  # NOTE: foreign load must come from OUTSIDE this process tree: the .NET
  # runtime subreaps orphans, so setsid/double-fork children stay lane-owned
  # (correctly excluded - builds/legs must never self-abort). Protocol: this
  # leg prints START-BURN-NOW, waits 25 s while the operator starts one CPU
  # burner from an independent session, measures, then prints STOP-BURN.
  Write-Host "START-BURN-NOW (one-CPU burner from an independent session)"
  Start-Sleep -Seconds 25
  $hot = Frac10
  Write-Host ("burn-10s frac=" + $hot.ToString("F3"))
  Check ($hot -gt 0.10) "deliberate foreign workload detected above abort line"
  Write-Host "STOP-BURN"
  $calm = Frac10
  Write-Host ("post-burn frac=" + $calm.ToString("F3"))
  Check ($calm -lt 0.02) "box calm again after burn exits"
}
if ($fail -gt 0) { Write-Host ("FAILURES: " + $fail); exit 1 }
Write-Host "ALL PROOF LEGS GREEN"
