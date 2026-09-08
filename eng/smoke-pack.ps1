param(
    [string]$Configuration = "Release"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Fail($msg) { Write-Host "FAIL smoke-pack: $msg"; exit 1 }

& dotnet pack src/Lokad.Onnx/Lokad.Onnx.csproj -c $Configuration --tl:off --nologo -v minimal | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "core pack failed" }
$nupkg = Get-ChildItem artifacts/nuget/Lokad.Onnx.*.nupkg | Where-Object { $_.Name -notlike "*.snupkg" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $nupkg) { Fail "no core nupkg found under artifacts/nuget" }
if ($nupkg.Name -notmatch "^Lokad\.Onnx\.(.+)\.nupkg$") { Fail ("unexpected package name: " + $nupkg.Name) }
$version = $Matches[1]
Write-Host "PASS pack"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($nupkg.FullName)
try {
    $names = @($zip.Entries | ForEach-Object { $_.FullName })
    foreach ($required in @("lib/net10.0/Lokad.Onnx.dll", "Lokad.Onnx.nuspec", "LICENSE.txt", "README.md", "CHANGELOG.md")) {
        if ($names -notcontains $required) { Fail ("package is missing " + $required) }
    }
    $nuspec = $zip.Entries | Where-Object { $_.Name.EndsWith(".nuspec") } | Select-Object -First 1
    $reader = New-Object System.IO.StreamReader($nuspec.Open())
    try { $spec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    if ($spec -match "<dependency[ />]") { Fail "package declares runtime dependencies" }
}
finally { $zip.Dispose() }
Write-Host "PASS contents"
$work = Join-Path $root "artifacts/smoke-pack/app"
$feed = Join-Path $root "artifacts/nuget"
if (Test-Path $work) { Fail "stale smoke workdir present, remove artifacts/smoke-pack first" }
New-Item -ItemType Directory -Force $work | Out-Null
Set-Content (Join-Path $work "nuget.config") "<configuration><packageSources><clear /></packageSources></configuration>"
Push-Location $work
try {
    & dotnet new console -f net10.0 --no-restore | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "scaffolding failed" }
    Remove-Item Program.cs
    $program = @(
'using System;',
'using System.Collections.Generic;',
'using Lokad.Onnx;',
'var mp = new OnnxModel { Name = "smoke" };',
'mp.Opset[""] = 11;',
'mp.Inputs.Add(new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = new[] { 2 } });',
'mp.Outputs.Add(new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = new[] { 2 } });',
'mp.Nodes.Add(new OnnxNode { Name = "r", OpType = "Relu", Inputs = new[] { "x" }, Outputs = new[] { "y" }, Attributes = new Dictionary<string, object>() });',
'var graph = Model.Load(mp);',
'var ok = graph.Execute(new Dictionary<string, ITensor> { { "x", DenseTensor<float>.OfValues(new float[] { -1f, 2f }) } }, true);',
'var y = ((Tensor<float>)graph.Outputs["y"]).ToArray();',
'Console.WriteLine("SMOKE-OK " + ok + " " + string.Join(",", y));'
    )
    Set-Content Program.cs ($program -join "`r`n")
    & dotnet add package Lokad.Onnx --version $version --source $feed | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "package reference failed" }
    $out = & dotnet run -c $Configuration --no-restore 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "scratch run failed" }
    if (-not ($out -match "SMOKE-OK True 0,2")) { Fail "unexpected scratch output" }
}
finally { Pop-Location }
Write-Host "PASS consume"
Write-Host "PASS smoke-pack"
