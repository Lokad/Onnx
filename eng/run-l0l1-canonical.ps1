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
. (Join-Path $PSScriptRoot "proc-acct.ps1")
# Tiering discipline (E02 root cause 2026-09-17): tiered JIT promotes and
# OSRs hot methods mid-series (deterministic 2-3x spikes, e.g. E5-8 iter 6
# on any box however quiet) and steady Tier0/OSR code runs ~1.75x slower
# than full opts. Release evidence pins full-opts code on BOTH legs via the
# inherited environment; the manifest env= line proves it (must read 0).
# Deployment-default tiering stays a separate diagnostic, never evidence.
$env:DOTNET_TieredCompilation = "0"
# Portable interpreter: dev boxes expose python, minimal Linux images only python3.
$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "python3" }
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
$l0want = (& git rev-parse $L0Ref 2>&1 | Out-String).Trim()
if ($l0want -notmatch "^[0-9a-f]{40}$") { Log ("ABORT: cannot resolve L0Ref " + $L0Ref); exit 1 }
$l0have = ""
if (Test-Path (Join-Path $l0root "Lokad.Onnx.slnx")) { $l0have = (& git -C $l0root rev-parse HEAD 2>&1 | Out-String).Trim() }
if ($l0have -ne $l0want) {
  if ($l0have -ne "") { Log ("stale L0 worktree " + $l0have + ", rebuilding at " + $l0want); & git worktree remove --force $l0root 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Encoding utf8 }
  & git worktree add --detach $l0root $l0want 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Append -Encoding utf8
  if ($LASTEXITCODE -ne 0) { Log "ABORT: L0 worktree failed, see worktree.log"; exit 1 }
}
Log ("L0-rev-verified=" + $l0want)
$l0models = Join-Path $l0root "models"
if (!(Test-Path $l0models)) {
  if ($IsLinux) { & ln -s (Join-Path $root "models") $l0models 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Append -Encoding utf8 }
  else {
    $t = "cmd /c mklink /J `"$l0models`" `"$(Join-Path $root 'models')`""
    Invoke-Expression $t 2>&1 | Out-File (Join-Path $runDir "worktree.log") -Append -Encoding utf8
  }
  if ($LASTEXITCODE -ne 0) { Log "ABORT: models junction failed"; exit 1 }
}
# Same-toolchain pairing (E01/D02): the 0.2.0 baseline pins a preview SDK that the
# quiet box neither installs nor needs - 0.2.0 source builds clean under the L1 pin
# (proven on the VM). The worktree is disposable; copy the L1 pin over it and let the
# L0-sdk manifest line prove the pairing every run.
Copy-Item (Join-Path $root "global.json") (Join-Path $l0root "global.json") -Force
Log "L0-toolchain-override=global.json-from-L1"
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
Log ("L1-sdk=" + ((& dotnet --info 2>&1 | Select-Object -First 4) -join " | "))
Push-Location $l0root
Log ("L0-sdk=" + ((& dotnet --info 2>&1 | Select-Object -First 4) -join " | "))
Pop-Location
Log ("runtimes=" + ((& dotnet --list-runtimes 2>&1 | Select-String "NETCore.App 10" | ForEach-Object { $_.Line.Trim() }) -join " | "))
Log ("env=" + (($env:DOTNET_EnableHWIntrinsic, $env:DOTNET_TieredCompilation, $env:DOTNET_JitOSR, $env:LOKAD_ONNX_GELU_TANH, $env:LOKAD_ONNX_SOFTMAX_SPAN) -join ","))
$ortpkg = (Select-String -Path tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -Pattern 'OnnxRuntime.*Version="([^"]+)"').Matches[0].Groups[1].Value
$ortNativeName = $(if ($IsLinux) { "libonnxruntime.so" } else { "onnxruntime.dll" })
$ortdll = Get-ChildItem tests/Lokad.Onnx.Bench/bin/Release/net10.0/runtimes/*/native/* -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq $ortNativeName } | Select-Object -First 1
if (-not $ortdll) { Log ("ABORT: loaded ORT native module missing (expected " + $ortNativeName + ")"); exit 1 }
$ortArch = $(if ($IsLinux) { (& uname -m 2>$null | Out-String).Trim() } else { $env:PROCESSOR_ARCHITECTURE })
Log ("ort-pkg=" + $ortpkg + " ort-native=" + $ortdll.FullName + " ort-native-sha=" + (Sha $ortdll.FullName) + " ort-arch=" + $ortArch)
$selfAff = "unreadable"
try { $selfAff = (Get-Process -Id $PID).ProcessorAffinity } catch { $selfAff = "unreadable:" + $_.Exception.Message }
Log ("affinity-requested=" + $Cpu + " self-affinity=" + $selfAff)
foreach ($m in @("models/multilingual-e5-small/model.onnx", "models/dinov3-vits16/onnx/model.onnx", "models/resnet50-onnx/model.onnx", "models/gpt2-onnx/onnx/model.onnx")) {
  Log ("asset " + $m + " sha=" + (Sha $m))
}
# Foreign-process snapshots: deltas and newcomers abort release evidence.


function Snap($phase) {
  if ($IsLinux) {
    Get-ProcTable | Where-Object { $_.Name -match "^(dotnet|testhost|python3?)$" } |
      Export-Csv (Join-Path $runDir ("foreign-" + $phase + ".csv")) -NoTypeInformation -Encoding utf8
  } else {
    Get-CimInstance Win32_Process | Where-Object { $_.Name -match "^(dotnet|testhost|python)(\.exe)?$" } |
      Select-Object Name, ProcessId, ParentProcessId, @{n="CPU";e={$_.KernelModeTime}}, CreationDate |
      Export-Csv (Join-Path $runDir ("foreign-" + $phase + ".csv")) -NoTypeInformation -Encoding utf8
  }
  Log ("foreign-snapshot " + $phase)
}
# Foreign-CPU accounting (E02 hardening): per-process CPU seconds attributed
# across each rep interval. (A per-processor busy-counter design was tried and
# retired: a 120s three-way calibration showed its busy+idle fractions summing
# to ~1.94x wall time on this box, so its 0.30 sibling line was unprincipled
# and the DryRun WATERMARK it produced is retracted as gate evidence. Snapshot
# plumbing from that attempt is retained.) TotalProcessorTime CPU-seconds are
# unambiguous. Newcomer PIDs count fully (conservative); processes that exit
# mid-rep are missed (documented; the steady gate backstops). PID 0 (Idle) is
# excluded outright or every interval reads ~0.8. Lane-owned PIDs (this script
# plus any descendant, so pinned legs, builds and the anchor) are excluded by
# ancestor walk and can never self-abort. A rep averaging above $ForeignAbort
# of box CPU-seconds aborts release evidence: sustained double-digit percent
# of the box over minutes cannot be OS jitter. Observed real contamination
# reads 0.12-0.16 here; an idle box reads ~0.02. Residual hole: a lone
# single-thread squatter reads ~1/28 and is missed; ABBA order plus the
# steady gate backstop that case.
$ForeignAbort = 0.10
$LogicalCpus = $(if ($IsLinux) { [int]((& nproc 2>$null | Out-String).Trim()) } else { (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors })
Log ("foreign-abort=" + $ForeignAbort + " logical-cpus=" + $LogicalCpus)
if ($IsLinux) {
  Log ("cpu-model=" + ((& lscpu 2>$null | Select-String "Model name" | ForEach-Object { $_.Line.Trim() }) -join " | "))
  Log ("cpu-siblings=" + (Get-Content ("/sys/devices/system/cpu/cpu" + $Cpu + "/topology/thread_siblings_list") -ErrorAction SilentlyContinue))
} else {
  Log ("cpu-model=" + ($env:PROCESSOR_IDENTIFIER) + " arch=" + ($env:PROCESSOR_ARCHITECTURE))
}
$script:prevProcSnap = $null

function SnapForeign($phase) {
  $map = @{}
  $rows = @()
  foreach ($pr in (Get-ProcTable)) {
    $map[$pr.Id] = @($pr.CPUSec, $pr.PPid)
    $rows += $pr
  }
  $now = (Get-Date).ToUniversalTime()
  $rows | Export-Csv (Join-Path $runDir ("proc-" + $phase + ".csv")) -NoTypeInformation -Encoding utf8
  $prev = $script:prevProcSnap
  $script:prevProcSnap = @{ Map = $map; Time = $now }
  if ($null -eq $prev) { Log ("foreign-cpu " + $phase + " baseline"); return -1.0 }
  $wall = ($now - $prev.Time).TotalSeconds
  $foreign = 0.0
  $new = 0
  foreach ($id in $map.Keys) {
    if ($id -eq 0) { continue }
    if (Test-LaneOwned $id $map) { continue }
    $c = $map[$id][0]
    if ($prev.Map.ContainsKey($id)) {
      $d = $c - $prev.Map[$id][0]
      if ($d -gt 0) { $foreign += $d }
    } else { $foreign += $c; $new++ }
  }
  if ($phase -match "(^pre$|-pre$)") { $script:ProcErrAtPre = $script:ProcParseErrors }
  $frac = 0.0
  if ($wall -gt 0 -and $LogicalCpus -gt 0) { $frac = $foreign / ($wall * $LogicalCpus) }
  Log ("foreign-cpu " + $phase + " frac=" + $frac.ToString("F2") + " new=" + $new)
  return $frac
}
Snap "pre"
$null = SnapForeign "pre"
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
  $null = SnapForeign ("rep" + $r + "-pre")
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
  $foreignFrac = SnapForeign ("rep" + $r + "-post")
  $parseBad = ($script:ProcParseErrors -gt $script:ProcErrAtPre)
  if (($foreignFrac -ge 0 -and $foreignFrac -gt $ForeignAbort) -or $parseBad) {
    $why = $(if ($parseBad) { "unparseable process accounting (errors=" + $script:ProcParseErrors + ")" } else { "foreign CPU " + $foreignFrac.ToString("F2") + " (sustained box-wide pressure)" })
    if ($DryRun) { Log ("WATERMARK: " + $why + " during rep " + $r + " would abort release evidence") }
    else { Log ("ABORT: " + $why + " during rep " + $r); exit 1 }
  }
}
Snap "post"
$null = SnapForeign "post"
& $py eng/score_campaign.py @l0logs --l1 @l1logs 2>&1 | Out-File (Join-Path $runDir "score.log") -Encoding utf8
$sec = $LASTEXITCODE
Get-Content (Join-Path $runDir "score.log") | ForEach-Object { Log ("score: " + $_) }
Log ("score-exit=" + $sec)
if ($sec -ne 0 -and -not $DryRun) { exit $sec }
Write-Host ("CAMPAIGN DONE: " + $runDir + " score-exit=" + $sec)