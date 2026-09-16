# R02 canonical re-run lane (frozen B01 protocol + earned methodology rules).
# Does NOT run here by accident: it executes three full canonical reps plus a
# BDN anchor (~30-45 min) and must only launch on a cool quiet box.
# Run: pwsh -NoProfile -File eng/rerun-canonical.ps1
# Rules encoded: foreign-dotnet check before AND after (abort unless -Force),
# spread/ratio anchor gate (eng/parse_anchor.ps1), fresh process per rep,
# inter-rep cooldown against asymmetric heat throttle, timestamped logs that
# NEVER overwrite artifacts/scoreboard-rep{1,2,3}.log, parser run at the end.
param([switch]$Force, [int]$Iters = 9, [int]$CooldownSeconds = 300)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$dll = "tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $root ("artifacts/rerun-" + $stamp)
New-Item -ItemType Directory -Force $runDir | Out-Null
$scriptStart = Get-Date
function ForeignDotnet() {
  Get-Process dotnet -ErrorAction SilentlyContinue |
    Where-Object { $_.StartTime -lt $scriptStart -and $_.CPU -gt 120 } |
    Select-Object Id, CPU, StartTime
}
function CheckForeign($phase) {
  $f = @(ForeignDotnet)
  $f | Out-File (Join-Path $runDir ("foreign-" + $phase + ".log")) -Encoding utf8
  if ($f.Count -gt 0 -and -not $Force) {
    Write-Host ("ABORT: foreign dotnet active at " + $phase + ":")
    $f | Format-Table -AutoSize | Out-String | Write-Host
    Write-Host "Re-run on a quiet box, or pass -Force to override (logged)."
    exit 1
  }
  Write-Host ("foreign check " + $phase + ": " + $f.Count + " candidate(s)")
}
"HEAD=$(git rev-parse --short HEAD) DLL=$((Get-Item $dll).LastWriteTime.ToString('o'))" | Out-File (Join-Path $runDir "manifest.log") -Encoding utf8
dotnet build tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -c Release --tl:off --nologo -v minimal 2>&1 | Out-File (Join-Path $runDir "build.log") -Encoding utf8
if ($LASTEXITCODE -ne 0) { Write-Host "ABORT: bench build failed, see build.log"; exit 1 }
CheckForeign "pre"
& dotnet $dll micro oneop --cpu 4 --filter "*Mm30Up*" 2>&1 | Out-File (Join-Path $runDir "anchor.log") -Encoding utf8
$gate = & pwsh -NoProfile -File (Join-Path $root "eng/parse_anchor.ps1") (Join-Path $runDir "anchor.log") 2>&1
$gate | Out-File (Join-Path $runDir "anchor-gate.log") -Encoding utf8
$gate | Out-File (Join-Path $runDir "manifest.log") -Append -Encoding utf8
if ($LASTEXITCODE -ne 0) { Write-Host "ABORT: anchor gate failed, see anchor-gate.log"; exit 1 }
Write-Host "anchor gate passed" 
function SnapshotThermal($phase) {
  # Telemetry only (see evidence 2026-09-15): idle turbo ~150% at ~5% utility;
  # sustained AVX2 sags with heat. Recorded per rep for post-hoc drift
  # analysis; the Mm30Up anchor stays the gate, this never aborts.
  try {
    $c = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation |
      Where-Object { $_.Name -eq "0,4" } |
      Select-Object -First 1
    if ($c) { ("thermal " + $phase + " cpu4 perfPct=" + $c.PercentProcessorPerformance + " utilPct=" + $c.PercentProcessorUtility) | Out-File (Join-Path $runDir "manifest.log") -Append -Encoding utf8 }
  } catch { ("thermal " + $phase + " unreadable: " + $_.Exception.Message) | Out-File (Join-Path $runDir "manifest.log") -Append -Encoding utf8 }
}
$repLogs = @()
foreach ($r in @(1,2,3)) {
  if ($r -gt 1) { Write-Host ("cooldown " + $CooldownSeconds + "s before rep " + $r); Start-Sleep -Seconds $CooldownSeconds }
  CheckForeign ("rep" + $r)
  SnapshotThermal ("rep" + $r)
  $log = Join-Path $runDir ("rep" + $r + ".log")
  $err = Join-Path $runDir ("rep" + $r + ".err.log")
  & dotnet $dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --rows canonical --cpu 4 --iters $Iters 1> $log 2> $err
  "rep${r}_exit=$LASTEXITCODE" | Out-File (Join-Path $runDir "manifest.log") -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Write-Host ("ABORT: rep " + $r + " failed, see " + $log); exit 1 }
  $repLogs += $log
}
CheckForeign "post"
python eng/parse_baseline.py $repLogs 2>&1 | Out-File (Join-Path $runDir "parser.log") -Encoding utf8
Write-Host ("RERUN DONE: " + $runDir)
