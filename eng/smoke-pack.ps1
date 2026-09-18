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
    foreach ($required in @("lib/net10.0/Lokad.Onnx.dll", "Lokad.Onnx.nuspec", "LICENSE.txt", "README.md", "CHANGELOG.md", "icon.png")) {
        if ($names -notcontains $required) { Fail ("package is missing " + $required) }
    }
    $nuspec = $zip.Entries | Where-Object { $_.Name.EndsWith(".nuspec") } | Select-Object -First 1
    $reader = New-Object System.IO.StreamReader($nuspec.Open())
    try { $spec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $depIds = @([regex]::Matches($spec, '<dependency id="([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
    # I03 contract: the integrated importer ships on exactly Google.Protobuf 3.33.5; nothing else, no OnnxSharp, no separate importer assembly.
    $unexpected = @($depIds | Where-Object { $_ -ne "Google.Protobuf" })
    if ($unexpected.Count -gt 0) { Fail ("package declares unexpected runtime dependencies: " + ($unexpected -join ",")) }
    $pbVer = [regex]::Match($spec, '<dependency id="Google.Protobuf" version="([^"]+)"').Groups[1].Value
    if ($depIds -notcontains "Google.Protobuf" -or $pbVer -ne "3.33.5") { Fail "package must declare exactly Google.Protobuf 3.33.5" }
    if ($names -contains "lib/net10.0/Lokad.Onnx.Import.dll") { Fail "package must not ship a separate importer assembly" }
    if ($spec -match "OnnxSharp") { Fail "package must not mention OnnxSharp" }
    $readmeEntry = $zip.Entries | Where-Object { $_.Name -eq "README.md" } | Select-Object -First 1
    $readmeReader = New-Object System.IO.StreamReader($readmeEntry.Open())
    try { $readmeText = $readmeReader.ReadToEnd() } finally { $readmeReader.Dispose() }
    if ($readmeText -match '\]\((?!https?://|#|mailto:)') { Fail "packaged README has repository-relative links" }
}
finally { $zip.Dispose() }
Write-Host "PASS contents"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$work = Join-Path $root ("artifacts/smoke-pack/app-" + $stamp)
$feed = Join-Path $work "feed"
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force $feed | Out-Null
Copy-Item $nupkg.FullName $feed
Set-Content (Join-Path $work "nuget.config") "<configuration><packageSources><clear /><add key=`"local`" value=`"feed`" /><add key=`"nuget.org`" value=`"https://api.nuget.org/v3/index.json`" /></packageSources></configuration>"
$previousPackages = $env:NUGET_PACKAGES
$env:NUGET_PACKAGES = Join-Path $work "packages"
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
'Console.WriteLine("SMOKE-OK " + ok + " " + string.Join(",", y));',
('var fixture = @"' + (Join-Path $root "tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx") + '";'),
'var dto = OnnxImport.Parse(fixture);',
'var inName = dto.Inputs[0].Name;',
'var fed = DenseTensor<float>.OfShape(dto.Inputs[0].Dims);',
'fed.Fill(0.5f);',
'var g1 = OnnxImport.Load(fixture)!;',
'var ok1 = g1.Execute(new Dictionary<string, ITensor> { { inName, fed } }, true);',
'var o1 = ((Tensor<float>)g1.Outputs.Values.First()).ToArray();',
'var g2 = OnnxImport.Load(File.ReadAllBytes(fixture))!;',
'var ok2 = g2.Execute(new Dictionary<string, ITensor> { { inName, fed } }, true);',
'var o2 = ((Tensor<float>)g2.Outputs.Values.First()).ToArray();',
'var odims = string.Join("x", ((Tensor<float>)g1.Outputs.Values.First()).Dimensions.ToArray());',
'Console.WriteLine("SMOKE-IMPORT-OK " + ok1 + " " + ok2 + " " + o1.SequenceEqual(o2) + " " + odims);',
'var romBytes = File.ReadAllBytes(fixture);',
'var romPadded = new byte[romBytes.Length + 64];',
'Array.Copy(romBytes, 0, romPadded, 37, romBytes.Length);',
'var romSlice = new ReadOnlyMemory<byte>(romPadded, 37, romBytes.Length);',
'var dtoRom = OnnxImport.Parse(romSlice);',
'var g3 = OnnxImport.Load(romSlice)!;',
'var ok3 = g3.Execute(new Dictionary<string, ITensor> { { inName, fed } }, true);',
'var o3 = ((Tensor<float>)g3.Outputs.Values.First()).ToArray();',
'Console.WriteLine("SMOKE-ROM-OK " + ok3 + " " + o1.SequenceEqual(o3) + " " + (dtoRom.Nodes.Count == dto.Nodes.Count));'
    )
    Set-Content Program.cs ($program -join "`r`n")
    & dotnet add package Lokad.Onnx --version $version | Out-Null
    if ($LASTEXITCODE -ne 0) { Fail "package reference failed" }
    $out = & dotnet run -c $Configuration --no-restore 2>&1
    if ($LASTEXITCODE -ne 0) { Fail "scratch run failed" }
    if (-not ($out -match "SMOKE-OK True 0,2")) { Fail "unexpected scratch output" }
    if (-not ($out -match "SMOKE-IMPORT-OK True True True 1x10")) { Fail "fixture import check failed" }
    if (-not ($out -match "SMOKE-ROM-OK True True True")) { Fail "ROM import check failed" }
}
finally {
    $env:NUGET_PACKAGES = $previousPackages
    Pop-Location
}
# Clean only the scratch directory created above; never mutate the user's
# global package cache to force the local package to be selected.
$smokeRoot = [System.IO.Path]::GetFullPath((Join-Path $root "artifacts/smoke-pack"))
$resolvedWork = [System.IO.Path]::GetFullPath($work)
$prefix = $smokeRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $resolvedWork.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    Fail "scratch cleanup path is outside artifacts/smoke-pack"
}
Remove-Item -LiteralPath $resolvedWork -Recurse -Force
Write-Host "PASS consume"
Write-Host "PASS smoke-pack"
