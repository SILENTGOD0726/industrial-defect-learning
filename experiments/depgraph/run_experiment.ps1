param(
    [ValidatePattern('^run_[A-Za-z0-9_-]+$')]
    [string]$RunName = 'run_2',
    [ValidateRange(1, 1000)]
    [int]$Epochs = 1
)
$ErrorActionPreference = 'Stop'
$experimentPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$experimentTemp = Join-Path $PSScriptRoot 'tmp'
$experimentOutput = Join-Path $PSScriptRoot $RunName
New-Item -ItemType Directory -Path $experimentTemp -Force | Out-Null
$env:TEMP = $experimentTemp
$env:TMP = $experimentTemp
$env:PYTHONIOENCODING = 'utf-8'
& $experimentPython (Join-Path $PSScriptRoot 'prepare_data.py')
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
& $experimentPython -u (Join-Path $PSScriptRoot 'experiment.py') --out_dir $experimentOutput --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; inspect the output above.' }
& $experimentPython -u (Join-Path $PSScriptRoot 'experiment.py') --out_dir $experimentOutput --reload-only
if ($LASTEXITCODE -ne 0) { throw 'Independent reload verification failed.' }
& $experimentPython (Join-Path $PSScriptRoot 'plot.py') --out_dir $experimentOutput
if ($LASTEXITCODE -ne 0) { throw 'Plot generation failed.' }
& $experimentPython (Join-Path $PSScriptRoot 'summarize.py') --out_dir $experimentOutput
if ($LASTEXITCODE -ne 0) { throw 'Learning note generation failed.' }
Write-Output "Completed: $experimentOutput"
