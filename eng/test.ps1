param(
    [string]$Configuration = "Release",
    [switch]$Coverage,
    [switch]$BuildTools,
    [ValidateSet('net10.0', 'net11.0')][string]$Framework
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
$buildSolution = if ($BuildTools) { 'Lokad.Onnx.slnx' } else { 'eng/Tests.slnx' }
Invoke-Dotnet @('restore', '--tl:off', '-v', 'minimal', $buildSolution)
$frameworkArguments = if ($Framework) { @('-f', $Framework) } else { @() }
Invoke-Dotnet (@('build', '--tl:off', '--nologo', '-v', 'minimal', '-c', $Configuration, $buildSolution, '--no-restore') + $frameworkArguments)
$testArguments = @('test', '--tl:off', '--nologo', '-v', 'minimal', '-c', $Configuration, '--no-build', '-p:TestTfmsInParallel=false', 'eng/Tests.slnx') + $frameworkArguments
if ($Coverage) { $testArguments += @('--collect:XPlat Code Coverage', '--results-directory', 'artifacts/coverage') }
Invoke-Dotnet $testArguments
Write-Host "PASS ordinary gate ($Configuration)"
