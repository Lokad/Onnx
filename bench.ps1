param(
  [int]$Iters = 3,
  [int]$MatMulIterations = 5,
  [int]$MatMulWarmups = 2,
  [string]$Model = 'models/multilingual-e5-small/model.onnx',
  [string]$OutDir = 'artifacts/bench'
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root
if (-not (Test-Path $Model)) { throw "Model not found: $Model (expected the git-ignored local e5 asset)." }
$cli = 'src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.CLI.exe'
dotnet build src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj -c Release --tl:off --nologo -v minimal -p:NuGetAudit=false | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'CLI Release build failed.' }
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
    $m = (Select-String -Path $log -Pattern 'Executing graph .* completed in (\d+)ms' | Select-Object -Last 1).Matches.Groups[1].Value
    $graphs += [int]$m; $walls += [int]$w
  }
  return @{ graphMs = $graphs; wallMs = $walls }
}

$short = 'query: hello world'
$long = 'query: The quick brown fox jumps over the lazy dog near the river bank in springtime weather for a pleasant afternoon walk'
$e5short = Run-E5 'short-8tok' $short
$e5long = Run-E5 'long-30tok' $long

$bdnLog = Join-Path $logDir 'matmul-bdn.log'
& $cli benchmark matmul --iterationCount $MatMulIterations --warmupCount $MatMulWarmups > $bdnLog 2>&1
if ($LASTEXITCODE -ne 0) { throw "matmul benchmark failed. See $bdnLog" }
$matmul = @{}
$csv = Get-ChildItem BenchmarkDotNet.Artifacts/results/*TensorMatMulBenchmarks-report.csv -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $csv) { throw "matmul benchmark produced no CSV report. See $bdnLog" }
Copy-Item $csv.FullName $logDir
foreach ($row in (Import-Csv $csv.FullName)) {
  $mm = [regex]::Match($row.Mean, '([\d,\.]+)\s*(ns|us|ms|s)')
  if ($mm.Success) {
    $v = [double]($mm.Groups[1].Value -replace ',', '')
    $u = $mm.Groups[2].Value
    if ($u -eq 'ns') { $v = $v / 1e6 } elseif ($u -eq 'us') { $v = $v / 1e3 } elseif ($u -eq 's') { $v = $v * 1e3 }
    $matmul[$row.Method] = [math]::Round($v, 4)
  }
}

$result = [ordered]@{
  date = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
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
}
$outFile = Join-Path $OutDir "bench-$stamp.json"
$result | ConvertTo-Json -Depth 5 | Set-Content $outFile
Write-Output "WROTE $outFile"
Write-Output ('e5 short graphMs=' + ($e5short.graphMs -join ',') + ' wallMs=' + ($e5short.wallMs -join ','))
Write-Output ('e5 long  graphMs=' + ($e5long.graphMs -join ',') + ' wallMs=' + ($e5long.wallMs -join ','))
foreach ($k in ($matmul.Keys | Sort-Object)) { Write-Output ("matmul ${k}=" + $matmul[$k] + 'ms') }
