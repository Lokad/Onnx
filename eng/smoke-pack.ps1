param(
    [string]$Configuration = "Release",
    [string]$Output = 'artifacts/nuget',
    [ValidateSet('net10.0', 'net11.0')][string[]]$Frameworks = @('net10.0', 'net11.0')
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Fail($msg) { Write-Host "FAIL smoke-pack: $msg"; exit 1 }

$packageOutput = [IO.Path]::GetFullPath($Output)
& dotnet pack src/Lokad.Onnx/Lokad.Onnx.csproj -c $Configuration -o $packageOutput --tl:off --nologo -v minimal | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "core pack failed" }
$nupkg = Get-ChildItem -LiteralPath $packageOutput -Filter 'Lokad.Onnx.*.nupkg' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $nupkg) { Fail "no core nupkg found under $packageOutput" }
if ($nupkg.Name -notmatch "^Lokad\.Onnx\.(.+)\.nupkg$") { Fail ("unexpected package name: " + $nupkg.Name) }
$version = $Matches[1]
Write-Host "PASS pack"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($nupkg.FullName)
$coreHashes = @{}
try {
    $names = @($zip.Entries | ForEach-Object { $_.FullName })
    foreach ($required in @("lib/net10.0/Lokad.Onnx.dll", "lib/net11.0/Lokad.Onnx.dll", "Lokad.Onnx.nuspec", "LICENSE.txt", "README.md", "CHANGELOG.md", "icon.png")) {
        if ($names -notcontains $required) { Fail ("package is missing " + $required) }
    }
    $nuspec = $zip.Entries | Where-Object { $_.Name.EndsWith(".nuspec") } | Select-Object -First 1
    $reader = New-Object System.IO.StreamReader($nuspec.Open())
    try { $spec = $reader.ReadToEnd() } finally { $reader.Dispose() }
    $depIds = @([regex]::Matches($spec, '<dependency id="([^"]+)"') | ForEach-Object { $_.Groups[1].Value })
    if ($depIds.Count -ne 0) { Fail ("core must have no runtime dependencies: " + ($depIds -join ",")) }
    $assemblies = @($names | Where-Object { $_ -like "lib/*.dll" -or $_ -like "lib/*/*.dll" })
    if ($assemblies.Count -ne 2 -or @($assemblies | Where-Object { $_ -notin @('lib/net10.0/Lokad.Onnx.dll', 'lib/net11.0/Lokad.Onnx.dll') }).Count -ne 0) {
        Fail 'package must contain exactly the .NET 10 and .NET 11 core assemblies'
    }
    foreach ($framework in @('net10.0', 'net11.0')) {
        $stream = $zip.GetEntry("lib/$framework/Lokad.Onnx.dll").Open()
        try { $coreHashes[$framework] = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)) }
        finally { $stream.Dispose() }
    }
    $readmeEntry = $zip.Entries | Where-Object { $_.Name -eq "README.md" } | Select-Object -First 1
    $readmeReader = New-Object System.IO.StreamReader($readmeEntry.Open())
    try { $readmeText = $readmeReader.ReadToEnd() } finally { $readmeReader.Dispose() }
    if ($readmeText -match '\]\((?!https?://|#|mailto:)') { Fail "packaged README has repository-relative links" }
}
finally { $zip.Dispose() }
Write-Host "PASS contents"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
$smokeRoot = Join-Path $packageOutput 'consumers'
$work = Join-Path $smokeRoot ("app-" + $stamp)
$feed = Join-Path $work "feed"
New-Item -ItemType Directory -Force $work | Out-Null
New-Item -ItemType Directory -Force $feed | Out-Null
Copy-Item $nupkg.FullName $feed
# Only the private local feed is available: the packed core must restore offline.
Set-Content (Join-Path $work "nuget.config") '<configuration><packageSources><clear /><add key="local" value="feed" /></packageSources></configuration>'
Copy-Item (Join-Path $root "tests/Lokad.Onnx.PackageConsumer/Consumer.csproj") $work
Copy-Item (Join-Path $root "tests/Lokad.Onnx.PackageConsumer/Program.cs") $work
$previousPackages = $env:NUGET_PACKAGES
$env:NUGET_PACKAGES = Join-Path $work "packages"
Push-Location $work
try {
    & dotnet restore Consumer.csproj --tl:off -v minimal "-p:CorePackageVersion=$version" -p:NuGetAudit=false
    if ($LASTEXITCODE -ne 0) { Fail "package-only restore failed" }
    $assets = Get-Content obj/project.assets.json -Raw | ConvertFrom-Json
    $libraries = @($assets.libraries.PSObject.Properties.Name)
    if ($libraries.Count -ne 1 -or $libraries[0] -ne "Lokad.Onnx/$version") { Fail "consumer restored unexpected dependencies" }
    foreach ($framework in $Frameworks) {
        & dotnet build Consumer.csproj -c $Configuration -f $framework --no-restore --tl:off --nologo -v minimal "-p:CorePackageVersion=$version"
        if ($LASTEXITCODE -ne 0) { Fail "package-only build failed for $framework" }
        $binaryDirectory = "bin/$Configuration/$framework"
        $out = & dotnet (Join-Path $binaryDirectory 'Lokad.Onnx.PackageConsumer.dll')
        if ($LASTEXITCODE -ne 0 -or -not ($out -match "SMOKE-CORE-OK")) { Fail "package-only consumer failed for $framework" }
        if (-not ($out -match ([regex]::Escape('.NET ' + $framework.Substring(3) + '.')))) {
            Fail "consumer ran on the wrong runtime for $framework"
        }
        if ((Get-FileHash -LiteralPath "$binaryDirectory/Lokad.Onnx.dll" -Algorithm SHA256).Hash -ne $coreHashes[$framework]) {
            Fail "consumer did not select the packed $framework core"
        }
        $out | Write-Host
        $runtime = Get-Content "$binaryDirectory/Lokad.Onnx.PackageConsumer.deps.json" -Raw | ConvertFrom-Json
        $runtimeLibraries = @($runtime.libraries.PSObject.Properties.Name)
        if ($runtimeLibraries.Count -ne 2 -or $runtimeLibraries -notcontains "Lokad.Onnx/$version") { Fail "consumer runtime has unexpected dependencies" }
        if (Test-Path "$binaryDirectory/Google.Protobuf.dll") { Fail "consumer contains Protobuf" }
        if (Test-Path "$binaryDirectory/Lokad.Onnx.Protobuf.dll") { Fail "consumer contains reference adapter" }
        Write-Host "PASS consume $framework"
    }
}
finally {
    $env:NUGET_PACKAGES = $previousPackages
    Pop-Location
}
# Clean only the scratch directory created above; never mutate the user's
# global package cache to force the local package to be selected.
$smokeRoot = [System.IO.Path]::GetFullPath($smokeRoot)
$resolvedWork = [System.IO.Path]::GetFullPath($work)
$prefix = $smokeRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $resolvedWork.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    Fail "scratch cleanup path is outside artifacts/smoke-pack"
}
Remove-Item -LiteralPath $resolvedWork -Recurse -Force
Write-Host "PASS consume"
Write-Host "PASS smoke-pack"
