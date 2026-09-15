# L0/L1/ORT release campaign lane (E01/E02 M1). Times the released library (L0,
# isolated origin/master worktree) and the candidate library (L1, working tree)
# side by side against matched ORT with alternating order, then scores with
# eng/score_campaign.py. Release evidence: no -Force, no -DryRun, quiet box.
# Engineering smoke: -DryRun shrinks counts and watermarks foreign findings.
# Run: pwsh -NoProfile -File eng/run-l0l1-canonical.ps1 [-Iters 33] [-Reps 3] [-DryRun]
param([int]$Iters = 33, [int]$Reps = 4, [int]$CooldownSeconds = 300,
      [string]$L0Ref = "origin/master", [int]$Cpu = 4, [switch]$DryRun)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if ($DryRun -and $Iters -eq 33) { $Iters = 2 }
if ($DryRun -and $Reps -eq 3) { $Reps = 1 }
if ($Iters -lt 1 -or $Reps -lt 1) { Write-Host "ABORT: bad counts"; exit 1 }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $root ("artifacts/campaign-" + $stamp)
New-Item -ItemType Directory -Force $runDir | Out-Null
$man = Join-Path $runDir "manifest.log"
function Log($s) { $s | Out-File $man -Append -Encoding utf8; Write-Host $s }
function Sha($p) { (Get-FileHash $p -Algorithm SHA256).Hash }
$laneStart = Get-Date
Log ("lane=start mode=" + ($DryRun ? "dryrun" : "release") + " iters=" + $Iters + " reps=" + $Reps + " cpu=" + $Cpu + " l0ref=" + $L0Ref)
Log ("lane-pid=" + $PID + " start=" + $laneStart.ToString("o"))
# L0 worktree (local objects only, never fetch) with shared model assets.
$l0root = Join-Path $root "artifacts/worktrees/l0"
if (!(Test-Path (Join-Path $l0root "Lokad.Onnx.slnx"))) {
  & git worktree add --detach $l0root $L0Ref 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Log "ABORT: L0 worktree failed, see worktree.log"; exit 1 }
}
$l0models = Join-Path $l0root "models"
if (!(Test-Path $l0models)) {
  $t = "cmd /c mklink /J `"$l0models`" `"$(Join-Path $root 'models')`""
  Invoke-Expression $t 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Log "ABORT: models junction failed"; exit 1 }
}
$l0rev = (& git -C $l0root rev-parse HEAD 2>&1 | Out-String).Trim()
$l1rev = (& git rev-parse HEAD 2>&1 | Out-String).Trim()
$l1diff = (& git diff HEAD --stat 2>&1 | Out-String).Trim()
Log ("L0-rev=" + $l0rev + " L1-rev=" + $l1rev)
if ($l1diff -ne "" -and -not $DryRun) { Log "ABORT: dirty working tree for release evidence"; exit 1 }
if ($l1diff -ne "") { Log "WATERMARK: dirty tree engineering run" }
# Builds with final binary hashes.
$l1dll = "tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll"
$l0dll = Join-Path $l0root $l1dll
& dotnet build tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -c Release --tl:off --nologo -v minimal 2>&1 | Out-File (Join-Path $runDir "build-l1.log") -Encoding utf8
if ($LASTEXITCODE -ne 0) { Log "ABORT: L1 bench build failed"; exit 1 }
Push-Location $l0root
& dotnet build tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -c Release --tl:off --nologo -v minimal 2>&1 | Out-File (Join-Path $runDir "build-l0.log") -Encoding utf8
$lec = $LASTEXITCODE
Pop-Location
if ($lec -ne 0) { Log "ABORT: L0 bench build failed"; exit 1 }
Log ("L1-bench-sha=" + (Sha $l1dll) + " L0-bench-sha=" + (Sha $l0dll))
$l1core = "src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll"
Log ("L1-core-sha=" + (Sha $l1core) + " L0-core-sha=" + (Sha (Join-Path $l0root $l1core)))
Log ("sdk=" + ((& dotnet --info 2>&1 | Select-Object -First 12) -join " | "))
Log ("env=" + (($env:DOTNET_EnableHWIntrinsic, $env:DOTNET_TieredCompilation, $env:DOTNET_JitOSR, $env:LOKAD_ONNX_GELU_TANH, $env:LOKAD_ONNX_SOFTMAX_SPAN) -join ","))
$ortpkg = (Select-String -Path tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -Pattern 'OnnxRuntime.*Version="([^"]+)"').Matches[0].Groups[1].Value
$ortdll = Get-ChildItem tests/Lokad.Onnx.Bench/bin/Release/net10.0/runtimes/win-x64/native/onnxruntime.dll -ErrorAction SilentlyContinue | Select-Object -First 1
Log ("ort-pkg=" + $ortpkg + " ort-dll-sha=" + ($ortdll ? (Sha $ortdll.FullName) : "missing"))
Log ("affinity-requested=" + $Cpu + " self-affinity=" + (Get-Process -Id $PID).ProcessorAffinity)
foreach ($m in @("models/multilingual-e5-small/model.onnx", "models/dinov3-vits16/onnx/model.onnx", "models/resnet50-onnx/model.onnx", "models/gpt2-onnx/onnx/model.onnx")) {
  Log ("asset " + $m + " sha=" + (Sha $m))
}
# Foreign-process snapshots: deltas and newcomers abort release evidence.
function Snap($phase) {
  Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^(dotnet|testhost|python)(\.exe)?$" } |
    Select-Object Name, ProcessId, ParentProcessId, @{n="CPU";e={$_.KernelModeTime}}, CreationDate |
    Export-Csv (Join-Path $runDir ("foreign-" + $phase + ".csv")) -NoTypeInformation -Encoding utf8
  Log ("foreign-snapshot " + $phase)
}
# SMT-sibling busy snapshots (E02 hardening): Bench is pinned to $Cpu, so
# sustained busy time on its SMT sibling is foreign pressure by construction.
# Raw cumulative counters need no sampling waits; per-CPU busy fractions are
# computed between consecutive snapshots. A rep averaging above $SibBusyAbort
# on the sibling aborts release evidence (contamination, not steady-noise):
# over a multi-minute rep that level cannot be OS jitter. Sibling defaults to
# the adjacent thread of the $Cpu pair (Cpu 4 -> 5 matches the observed mask
# 0x30); the full per-CPU table stays in cpu-<phase>.csv for audit either way.
$SibBusyAbort = 0.30
$SibCpu = if ($Cpu % 2 -eq 0) { $Cpu + 1 } else { $Cpu - 1 }
Log ("sibling-cpu=" + $SibCpu + " sib-busy-abort=" + $SibBusyAbort)
$script:prevCpuSnap = $null
function SnapCpu($phase) {
  $cur = Get-CimInstance Win32_PerfRawData_PerfOS_Processor |
    Where-Object { $_.Name -ne "_Total" } | Sort-Object { [int]$_.Name } |
    Select-Object Name, PercentProcessorTime, TimeStamp_Sys100NS
  $cur | Export-Csv (Join-Path $runDir ("cpu-" + $phase + ".csv")) -NoTypeInformation -Encoding utf8
  $prev = $script:prevCpuSnap
  $script:prevCpuSnap = $cur
  if ($null -eq $prev) { Log ("cpu-snapshot " + $phase + " baseline"); return -1.0 }
  $parts = @()
  $sib = -1.0
  foreach ($c in $cur) {
    $p = $prev | Where-Object { $_.Name -eq $c.Name } | Select-Object -First 1
    if ($null -eq $p) { continue }
    $dt = [double]$c.TimeStamp_Sys100NS - [double]$p.TimeStamp_Sys100NS
    $busy = 0.0
    if ($dt -gt 0) {
      $busy = ([double]$c.PercentProcessorTime - [double]$p.PercentProcessorTime) / $dt
      if ($busy -lt 0) { $busy = 0 }
      if ($busy -gt 1) { $busy = 1 }
    }
    $parts += ($c.Name + "=" + $busy.ToString("F2"))
    if ([int]$c.Name -eq $SibCpu) { $sib = $busy }
  }
  Log ("cpu-busy " + $phase + " sib=" + $SibCpu + " " + ($parts -join " "))
  return $sib
}
Snap "pre"
SnapCpu "pre"
# Anchor on the L1 binary (box-state probe, library-independent).
& dotnet (Join-Path $root $l1dll) micro oneop --cpu $Cpu --artifacts (Join-Path $runDir "bdn-anchor") --filter "*Mm30Up*" 2>&1 | Out-File (Join-Path $runDir "anchor.log") -Encoding utf8
$gate = & pwsh -NoProfile -File (Join-Path $root "eng/parse_anchor2.ps1") (Join-Path $runDir "anchor.log") 2>&1
$gate | Out-File (Join-Path $runDir "anchor-gate.log") -Encoding utf8
$gate | ForEach-Object { Log ("anchor: " + $_) }
if ($LASTEXITCODE -ne 0) { Log "ABORT: anchor gate failed (contaminated box)"; exit 1 }
$l0logs = @(); $l1logs = @()
for ($r = 1; $r -le $Reps; $r++) {
  if ($r -gt 1) { Log ("cooldown " + $CooldownSeconds + "s before rep " + $r); Start-Sleep -Seconds $CooldownSeconds }
  Snap ("rep" + $r + "-pre")
  SnapCpu ("rep" + $r + "-pre")
  $l0first = if ($r -le [Math]::Ceiling($Reps / 2.0)) { $r % 2 -eq 1 } else { $r % 2 -eq 0 }
  $order = ($l0first) ? @("L0", "L1") : @("L1", "L0")
  Log ("rep" + $r + " order=" + ($order -join ","))
  foreach ($leg in $order) {
    $dll = ($leg -eq "L0") ? $l0dll : (Join-Path $root $l1dll)
    $log = Join-Path $runDir (($leg.ToLower()) + "-rep" + $r + ".log")
    $err = Join-Path $runDir (($leg.ToLower()) + "-rep" + $r + ".err.log")
    & dotnet $dll e5 dinov3 resnet50 gpt2 --mode auto --threads 1 --rows canonical --cpu $Cpu --iters $Iters 1> $log 2> $err
    $ec = $LASTEXITCODE
    Log ("rep" + $r + "-" + $leg + " exit=" + $ec)
    if ($ec -ne 0) { Log ("ABORT: rep failed, see " + $log); exit 1 }
    if ($leg -eq "L0") { $l0logs += $log } else { $l1logs += $log }
  }
  Snap ("rep" + $r + "-post")
  $sibBusy = SnapCpu ("rep" + $r + "-post")
  if ($sibBusy -ge 0 -and $sibBusy -gt $SibBusyAbort) {
    if ($DryRun) { Log ("WATERMARK: sibling CPU " + $SibCpu + " busy " + $sibBusy.ToString("F2") + " during rep " + $r + " would abort release evidence") }
    else { Log ("ABORT: sibling CPU " + $SibCpu + " busy " + $sibBusy.ToString("F2") + " during rep " + $r + " (foreign SMT pressure)"); exit 1 }
  }
}
Snap "post"
SnapCpu "post"
& python eng/score_campaign.py @l0logs --l1 @l1logs 2>&1 | Out-File (Join-Path $runDir "score.log") -Encoding utf8
$sec = $LASTEXITCODE
Get-Content (Join-Path $runDir "score.log") | ForEach-Object { Log ("score: " + $_) }
Log ("score-exit=" + $sec)
if ($sec -ne 0 -and -not $DryRun) { exit $sec }
Write-Host ("CAMPAIGN DONE: " + $runDir + " score-exit=" + $sec)