param([string]$TracePath, [string]$LibraryDirectory, [string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Existing output' }
[void][System.IO.Directory]::CreateDirectory($OutputDirectory)
foreach ($name in @('Microsoft.Diagnostics.FastSerialization.dll', 'Microsoft.Diagnostics.NETCore.Client.dll', 'Microsoft.Diagnostics.Tracing.TraceEvent.dll')) {
    [void][System.Reflection.Assembly]::LoadFrom((Join-Path $LibraryDirectory $name))
}
$library = [Microsoft.Diagnostics.Tracing.Etlx.TraceLog].Assembly
if ($library.GetName().Version.ToString() -ne '3.1.23.0') { throw 'Unexpected parser version' }
$options = [Microsoft.Diagnostics.Tracing.Etlx.TraceLogOptions]::new()
$options.KeepAllEvents = $true
$options.ContinueOnError = $false
$options.LocalSymbolsOnly = $true
$options.ShouldResolveSymbols = { param($path) return $false }
$options.OnLostEvents = { param($truncated, $lost, $count) throw "Incomplete conversion: truncated=$truncated lost=$lost events=$count" }
$options.ConversionLogName = Join-Path $OutputDirectory 'conversion.log'
$etlx = Join-Path $OutputDirectory 'trace.etlx'
[void][Microsoft.Diagnostics.Tracing.Etlx.TraceLog]::CreateFromEventPipeDataFile($TracePath, $etlx, $options)
$log = [Microsoft.Diagnostics.Tracing.Etlx.TraceLog]::new($etlx)
function Open-GzipWriter([string]$name) {
    $file = [System.IO.File]::Open((Join-Path $OutputDirectory $name), [System.IO.FileMode]::CreateNew)
    $gzip = [System.IO.Compression.GZipStream]::new($file, [System.IO.Compression.CompressionLevel]::Fastest)
    return [System.IO.StreamWriter]::new($gzip, [System.Text.UTF8Encoding]::new($false))
}
function Write-Row($writer, $row) { $writer.WriteLine(($row | ConvertTo-Json -Depth 6 -Compress)) }
$addresses = Open-GzipWriter 'addresses.jsonl.gz'
$stacks = Open-GzipWriter 'stacks.jsonl.gz'
$samples = Open-GzipWriter 'samples.jsonl.gz'
$boundaries = Open-GzipWriter 'boundaries.jsonl.gz'
$counts = @{ addresses = 0; stacks = 0; samples = 0; boundaries = 0; missingStacks = 0; allEvents = 0 }
try {
    for ($index = 0; $index -lt $log.CodeAddresses.Count; $index++) {
        $address = $log.CodeAddresses[[Microsoft.Diagnostics.Tracing.Etlx.CodeAddressIndex]$index]
        Write-Row $addresses ([ordered]@{
            index = [int]$address.CodeAddressIndex; address = ('0x{0:x}' -f $address.Address)
            method = $address.FullMethodName; module = $address.ModuleFilePath
            tier = $log.CodeAddresses.OptimizationTier($address.CodeAddressIndex).ToString()
            token = $(if ($null -eq $address.Method) { $null } else { $address.Method.MethodToken })
        })
        $counts.addresses++
    }
    for ($index = 0; $index -lt $log.CallStacks.Count; $index++) {
        $stack = $log.CallStacks[[Microsoft.Diagnostics.Tracing.Etlx.CallStackIndex]$index]
        Write-Row $stacks ([ordered]@{
            index = [int]$stack.CallStackIndex; address = [int]$stack.CodeAddress.CodeAddressIndex
            caller = $(if ($null -eq $stack.Caller) { -1 } else { [int]$stack.Caller.CallStackIndex })
        })
        $counts.stacks++
    }
    foreach ($event in $log.Events) {
        $counts.allEvents++
        if ($event.ProviderName -eq 'Microsoft-DotNETCore-SampleProfiler' -and [int]$event.ID -eq 0) {
            $stackIndex = [int]$log.GetCallStackIndexForEvent($event)
            if ($stackIndex -lt 0) { $counts.missingStacks++ }
            Write-Row $samples ([ordered]@{
                index = $counts.samples; eventIndex = [int]$event.EventIndex
                ms = $event.TimeStampRelativeMSec; pid = $event.ProcessID; thread = $event.ThreadID
                stack = $stackIndex; raw = [Convert]::ToBase64String($event.EventData())
            })
            $counts.samples++
        } elseif ($event.ProviderName -eq 'Lokad-Pyannote-Diagnostic') {
            Write-Row $boundaries ([ordered]@{
                ms = $event.TimeStampRelativeMSec; pid = $event.ProcessID; thread = $event.ThreadID
                id = [int]$event.ID; raw = [Convert]::ToBase64String($event.EventData())
            })
            $counts.boundaries++
        }
    }
    if ($log.EventsLost -ne 0 -or $counts.samples -ne 115795 -or $counts.boundaries -ne 120) { throw 'Incomplete event coverage' }
    $summary = [ordered]@{
        passed = $true; counts = $counts; lost = $log.EventsLost
        input_sha256 = (Get-FileHash -LiteralPath $TracePath -Algorithm SHA256).Hash.ToLowerInvariant()
        library_sha256 = (Get-FileHash -LiteralPath $library.Location -Algorithm SHA256).Hash.ToLowerInvariant()
        runtime = [Environment]::Version.ToString(); powershell = $PSVersionTable.PSVersion.ToString()
        pid = $PID; inference_calls = 0; symbol_downloads = $false
    }
    $summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'summary.json') -Encoding utf8NoBOM
} finally {
    $addresses.Dispose(); $stacks.Dispose(); $samples.Dispose(); $boundaries.Dispose(); $log.Dispose()
}
$summary | ConvertTo-Json -Depth 6 -Compress
