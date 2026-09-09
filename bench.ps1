param(
  [int]$Iters = 3,
  [int]$MatMulIterations = 5,
  [int]$MatMulWarmups = 2,
  [string]$Model = 'models/multilingual-e5-small/model.onnx',
  [string]$OutDir = 'artifacts/bench'
)
# Methodology (startup-inclusive): every e5 sample below runs in a FRESH CLI
# process, so per-sample wall time includes process startup, model load, and
# inference. The single warmup process warms only itself; it does not warm the
# later timed processes. Do not read these walls as warmed in-process throughput.
$ErrorActionPreference = 'Stop'
if ($Iters -lt 1) { throw "Iters must be at least 1 (got $Iters); empty runs cannot publish." }
$root = $PSScriptRoot
Set-Location $root
if (-not (Test-Path $Model)) { throw "Model not found: $Model (expected the git-ignored local e5 asset)." }
$cli = 'src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.CLI.exe'
$bench = 'tests/Lokad.Onnx.Bench/bin/Release/net10.0/Lokad.Onnx.Bench.dll'
dotnet build src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj -c Release --tl:off --nologo -v minimal -p:NuGetAudit=false | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'CLI Release build failed.' }
dotnet build tests/Lokad.Onnx.Bench/Lokad.Onnx.Bench.csproj -c Release --tl:off --nologo -v minimal -p:NuGetAudit=false | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Bench Release build failed.' }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logDir = Join-Path $OutDir "bench-$stamp-logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

function Run-E5([string]$name, [string]$text) {
  $graphs = @(); $walls = @()
  $log = Join-Path $logDir "e5-$name-warmup.log"
  & $cli run $Model $text --text me5s > $log 2>&1
  if ($LASTEXITCODE -ne 0) { throw "e5 warmup failed ($name). See $log" }
  for ($i = 1; $i -le $Iters; $i++) {
    $log = Join-Path $logDir "e5-$name-$i.log"
    $w = (Measure-Command { & $cli run $Model $text --text me5s > $log 2>&1 }).TotalMilliseconds
    if ($LASTEXITCODE -ne 0) { throw "e5 run failed ($name #$i). See $log" }
    # >>graphmatch-begin (exact lines under synthetic-fixture test in B07 plan)
    $hit = (Select-String -Path $log -Pattern 'Executing graph .* completed in (\d+)ms' | Select-Object -Last 1)
    if (-not $hit) { throw "e5 run produced no graph-time line ($name #$i). See $log" }
    $m = $hit.Matches.Groups[1].Value
    # >>graphmatch-end
    $graphs += [int]$m; $walls += [int]$w
  }
  return @{ graphMs = $graphs; wallMs = $walls }
}

$short = 'query: hello world'
$long = 'query: The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk'
$e5short = Run-E5 'short-8tok' $short
$e5long = Run-E5 'long-30tok' $long

# BenchmarkDotNet artifacts are isolated per invocation: only CSVs written by
# THIS run (under $bdnArtifacts) may supply its numbers. A stale report from an
# older run in the shared BenchmarkDotNet.Artifacts tree can never be picked up.
$bdnArtifacts = Join-Path $logDir 'bdn-artifacts'
New-Item -ItemType Directory -Force $bdnArtifacts | Out-Null
$bdnLog = Join-Path $logDir 'matmul-bdn.log'
& dotnet $bench micro matmul --artifacts $bdnArtifacts --iterationCount $MatMulIterations --warmupCount $MatMulWarmups > $bdnLog 2>&1
if ($LASTEXITCODE -ne 0) { throw "matmul benchmark failed. See $bdnLog" }
# >>bdnparse-begin (exact lines under synthetic-fixture test in B07 plan)
$csvs = @(Get-ChildItem (Join-Path $bdnArtifacts 'results') -Filter '*TensorMatMulBenchmarks-report.csv' -ErrorAction SilentlyContinue)
if ($csvs.Count -eq 0) { throw "matmul benchmark produced no CSV report in isolated dir $bdnArtifacts. See $bdnLog" }
$csv = ($csvs | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
Copy-Item $csv.FullName $logDir
$matmul = [ordered]@{}
$rows = @(Import-Csv $csv.FullName)
if ($rows.Count -eq 0) { throw "matmul CSV report has no data rows: $($csv.FullName). See $bdnLog" }
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
foreach ($row in $rows) {
  $mm = [regex]::Match($row.Mean, '([\d,\.\s]+)\s*(ns|us|ms|s)')
  if (-not $mm.Success) { throw "matmul CSV row has no parseable Mean: Method='$($row.Method)' Mean='$($row.Mean)'. See $bdnLog" }
  $numText = ($mm.Groups[1].Value -replace '[,\s]', '')
  try { $v = [double]::Parse($numText, $invariant) }
  catch { throw "matmul CSV Mean is not a number: Method='$($row.Method)' Mean='$($row.Mean)'. See $bdnLog" }
  $u = $mm.Groups[2].Value
  if ($u -eq 'ns') { $v = $v / 1e6 } elseif ($u -eq 'us') { $v = $v / 1e3 } elseif ($u -eq 's') { $v = $v * 1e3 }
  $matmul[$row.Method] = [math]::Round($v, 4)
}
if ($matmul.Count -eq 0) { throw "matmul CSV yielded no parsed methods: $($csv.FullName). See $bdnLog" }
# >>bdnparse-end

$methodology = 'startup-inclusive: fresh CLI process per e5 sample (walls include startup+load+inference); one warmup process does not warm later processes; matmul via BenchmarkDotNet with per-run isolated artifacts'
$result = [ordered]@{
  date = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
  methodology = $methodology
  commit = (git rev-parse --short HEAD)
  sdk = (dotnet --version)
  cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
  ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
  config = 'Release'
  e5 = [ordered]@{
    short8tok = $e5short
    long30tok = $e5long
  }
  matmulMeanMs = $matmul
  validation = [ordered]@{
    e5short8tok = [ordered]@{ runsOk = $e5short.graphMs.Count; runsRequested = $Iters; status = 'every CLI run exited 0 with a graph-time line' }
    e5long30tok = [ordered]@{ runsOk = $e5long.graphMs.Count; runsRequested = $Iters; status = 'every CLI run exited 0 with a graph-time line' }
    matmul = [ordered]@{ methodsParsed = $matmul.Count; csv = $csv.Name; status = 'BDN exit 0 with a fresh isolated CSV; cleanup agreement failures would have failed the run' }
  }
}
$outFile = Join-Path $OutDir "bench-$stamp.json"
$result | ConvertTo-Json -Depth 5 | Set-Content $outFile
Write-Output "WROTE $outFile"
Write-Output $methodology
Write-Output ('e5 short graphMs=' + ($e5short.graphMs -join ',') + ' wallMs=' + ($e5short.wallMs -join ','))
Write-Output ('e5 long  graphMs=' + ($e5long.graphMs -join ',') + ' wallMs=' + ($e5long.wallMs -join ','))
Write-Output ('validation: e5 short ' + $e5short.graphMs.Count + '/' + $Iters + ' ok; e5 long ' + $e5long.graphMs.Count + '/' + $Iters + ' ok; matmul ' + $matmul.Count + ' methods parsed from ' + $csv.Name)
foreach ($k in ($matmul.Keys | Sort-Object)) { Write-Output ("matmul ${k}=" + $matmul[$k] + 'ms') }
