param(
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = "pandapower-vpp-dso-sim\outputs\paper_training_long_$Stamp"
}

$ResolvedOutput = Join-Path $Root $OutputDir
$LogDir = Join-Path $ResolvedOutput "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$StatusLog = Join-Path $LogDir "paper_long_status.log"
$StdoutLog = Join-Path $LogDir "paper_long_stdout.log"
$StderrLog = Join-Path $LogDir "paper_long_stderr.log"
$CombinedLog = Join-Path $LogDir "paper_long_combined.log"

"started_at=$(Get-Date -Format o)" | Out-File -FilePath $StatusLog -Encoding utf8
"root=$Root" | Out-File -FilePath $StatusLog -Encoding utf8 -Append
"output_dir=$ResolvedOutput" | Out-File -FilePath $StatusLog -Encoding utf8 -Append

try {
    $PythonArgs = "pandapower-vpp-dso-sim\examples\17_paper_training_experiment.py --preset paper_long --output-dir `"$OutputDir`" --checkpoint-selection both"
    $Process = Start-Process -FilePath "python" `
        -ArgumentList $PythonArgs `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru `
        -WindowStyle Hidden `
        -Wait
    $ExitCode = $Process.ExitCode
    if (Test-Path $StdoutLog) {
        Get-Content $StdoutLog | Out-File -FilePath $CombinedLog -Encoding utf8 -Append
    }
    if (Test-Path $StderrLog) {
        Get-Content $StderrLog | Out-File -FilePath $CombinedLog -Encoding utf8 -Append
    }
    "finished_at=$(Get-Date -Format o)" | Out-File -FilePath $StatusLog -Encoding utf8 -Append
    "exit_code=$ExitCode" | Out-File -FilePath $StatusLog -Encoding utf8 -Append
    exit $ExitCode
}
catch {
    "failed_at=$(Get-Date -Format o)" | Out-File -FilePath $StatusLog -Encoding utf8 -Append
    "error=$($_.Exception.Message)" | Out-File -FilePath $StatusLog -Encoding utf8 -Append
    throw
}
