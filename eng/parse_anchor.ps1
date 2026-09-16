# Anchor-gate parser for the canonical re-run lane.
# Usage: pwsh -NoProfile -File eng/parse_anchor.ps1 <anchor.log>
# Exit 0 (LAUNCH) only when: both Lokad and Ort Mm30Up summaries parse,
# each (Max-Min)/Median spread is below 30 percent, and the Lokad/Ort
# median ratio sits inside 1.0-1.6. Everything else ABORTs with a reason.
# Rationale (evidence 2026-09-15): single-sample kernel absolutes swing +-11%
# across box states, wider than any fittable band; spreads catch contamination
# (probe1 ORT 44 percent) and machine-state drift (probe2 Lokad 37 percent)
# while ratios cancel common-mode drift. Thresholds are provisional, set from
# four anchor logs; the reps carry the precision verdict, never this gate.
param([string]$Log)
if (-not $Log -or -not (Test-Path $Log)) { Write-Host "ABORT: missing log"; exit 1 }
$stats = @()
foreach ($line in (Get-Content $Log)) {
  $m = [regex]::Match($line, "Min = ([0-9.]+) us, Q1 = [0-9.]+ us, Median = ([0-9.]+) us, Q3 = [0-9.]+ us, Max = ([0-9.]+) us")
  if ($m.Success) {
    $key = $m.Groups[0].Value
    if (-not ($stats | Where-Object { $_["key"] -eq $key })) {
      $stats += @(@{ "key" = $key; "min" = [double]$m.Groups[1].Value; "med" = [double]$m.Groups[2].Value; "max" = [double]$m.Groups[3].Value })
    }
  }
}
$table = Get-Content $Log | Where-Object { $_ -match "^\| .*- (Lokad|ORT) session" }
if ($stats.Count -lt 2) { Write-Host "ABORT: need both Lokad and Ort summaries"; exit 1 }
if (($table | Measure-Object).Count -lt 2) { Write-Host "ABORT: summary table missing a method"; exit 1 }
$lok = $stats[0]; $ort = $stats[1]
$lokSpread = ($lok["max"] - $lok["min"]) / $lok["med"]
$ortSpread = ($ort["max"] - $ort["min"]) / $ort["med"]
$ratio = $lok["med"] / $ort["med"]
Write-Host ("lokad median=" + $lok["med"] + " spread=" + $lokSpread.ToString("P1") + " ort median=" + $ort["med"] + " spread=" + $ortSpread.ToString("P1") + " ratio=" + $ratio.ToString("F3"))
if ($lokSpread -ge 0.30) { Write-Host "ABORT: lokad spread too wide"; exit 1 }
if ($ortSpread -ge 0.30) { Write-Host "ABORT: ort spread too wide"; exit 1 }
if ($ratio -lt 1.0 -or $ratio -gt 1.6) { Write-Host "ABORT: ratio outside 1.0-1.6 (also catches method swap)"; exit 1 }
Write-Host "LAUNCH"
exit 0
