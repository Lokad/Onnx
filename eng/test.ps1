param(
    [string]$Configuration = "Release"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Invoke-Step($cmd) {
    Write-Host ">> $cmd"
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        Invoke-Expression $cmd
    } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "Step failed ($LASTEXITCODE): $cmd" }
}
Invoke-Step "dotnet restore --tl:off -v minimal Lokad.Onnx.slnx"
Invoke-Step "dotnet build --tl:off --nologo -v minimal -c $Configuration Lokad.Onnx.slnx --no-restore"
Invoke-Step "dotnet test --tl:off --nologo -v minimal -c $Configuration --no-build Lokad.Onnx.slnx --collect:`"XPlat Code Coverage`" --results-directory artifacts/coverage"
Write-Host "PASS ordinary gate ($Configuration)"
