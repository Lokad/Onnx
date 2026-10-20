param(
    [string]$Configuration = "Release",
    [switch]$Coverage
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
function Invoke-Dotnet([string[]]$Arguments) {
    Write-Host ">> dotnet $($Arguments -join ' ')"
    $prev = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & dotnet @Arguments
    } finally { $ErrorActionPreference = $prev }
    if ($LASTEXITCODE -ne 0) { throw "dotnet failed ($LASTEXITCODE): $($Arguments -join ' ')" }
}
Invoke-Dotnet @('restore', '--tl:off', '-v', 'minimal', 'Lokad.Onnx.slnx')
Invoke-Dotnet @('build', '--tl:off', '--nologo', '-v', 'minimal', '-c', $Configuration, 'Lokad.Onnx.slnx', '--no-restore')
$testArguments = @('test', '--tl:off', '--nologo', '-v', 'minimal', '-c', $Configuration, '--no-build', 'Lokad.Onnx.slnx')
if ($Coverage) { $testArguments += @('--collect:XPlat Code Coverage', '--results-directory', 'artifacts/coverage') }
Invoke-Dotnet $testArguments
Write-Host "PASS ordinary gate ($Configuration)"
