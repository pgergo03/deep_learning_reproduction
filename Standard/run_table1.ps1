param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 4294967295)]
    [uint32]$Seed
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$python = Join-Path $root '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python virtual environment not found: $python"
}

$limit = if ($env:JOBS) { [int]$env:JOBS } else { 3 }
if ($limit -lt 1) { throw 'JOBS must be a positive integer' }
$env:PYTHONHASHSEED = $Seed.ToString()
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
foreach ($directory in 'Models', 'Results', 'mult_adv', 'CW_adv', 'logs') {
    New-Item -ItemType Directory -Path (Join-Path $root $directory) -Force |
        Out-Null
}

# name | train arguments | evaluation arguments
$variants = @(
    'std|--preproc 0|--preproc 0 --bil_m 10 --time std'
    'bil|--preproc 1 --bil 10 --noise 0|--preproc 1 --bil 10 --noise 0 --bil_m 10 --time bil'
    'noise|--preproc 1 --bil 0 --noise 0.032|--preproc 1 --bil 0 --noise 0.032 --bil_m 0 --time noise'
    'nb|--preproc 1|--preproc 1 --bil_m 10 --time nb'
    'bn|--preproc 1 --noise_first 0|--preproc 1 --noise_first 0 --bil_m 10 --time bn'
)

$trainTasks = @()
$evalTasks = @()
foreach ($variant in $variants) {
    $name, $trainArgs, $evalArgs = $variant -split '\|'
    $runTag = "${name}_seed_$Seed"
    $evalArgs = $evalArgs -replace '--time \S+', "--time $runTag"
    $modelPattern = Join-Path $root "Models\model_*_$runTag"
    if (-not (Get-ChildItem -Path $modelPattern -ErrorAction SilentlyContinue |
        Select-Object -First 1)) {
        $trainTasks += [pscustomobject]@{
            Script = 'train.py'; Arguments = "$trainArgs --tag $runTag --seed $Seed"
            Log = Join-Path $root "logs\train_$runTag.log"
        }
    }

    $evalTasks += [pscustomobject]@{
        Script = 'mult_adv.py'; Arguments = "$evalArgs --eps 0 0.03 --seed $Seed --run_tag $name"
        Log = Join-Path $root "logs\mult_$runTag.log"
    }
    $evalTasks += [pscustomobject]@{
        Script = 'CW_adv.py'; Arguments = "$evalArgs --seed $Seed --run_tag $name"
        Log = Join-Path $root "logs\cw_$runTag.log"
    }
}

function Complete-Task($job) {
    Wait-Job $job | Out-Null
    $code = Receive-Job $job -ErrorAction SilentlyContinue | Select-Object -Last 1
    $success = $job.State -eq 'Completed' -and $code -eq 0
    Remove-Job $job
    if (-not $success) { [Console]::Error.WriteLine("Task failed; see $($job.Name)") }
    return $success
}

function Invoke-Tasks($tasks) {
    $running = @()
    $success = $true
    foreach ($task in $tasks) {
        $running += Start-Job -Name $task.Log -ArgumentList @(
            $root, $python, $task.Script, $task.Arguments, $task.Log
        ) -ScriptBlock {
            param($root, $python, $pythonScript, $arguments, $log)
            Set-Location $root
            $pythonArgs = $arguments -split ' '
            & $python -u $pythonScript @pythonArgs *> $log
            [int]$LASTEXITCODE
        }
        if ($running.Count -lt $limit) { continue }
        $done = Wait-Job $running -Any
        if (-not (Complete-Task $done)) { $success = $false }
        $running = @($running | Where-Object Id -ne $done.Id)
    }
    foreach ($job in $running) {
        if (-not (Complete-Task $job)) { $success = $false }
    }
    return $success
}

if (-not (Invoke-Tasks $trainTasks)) { exit 1 }
if (-not (Invoke-Tasks $evalTasks)) { exit 1 }
Set-Location $root
& $python -u summarize_table1.py --seed $Seed
exit $LASTEXITCODE
