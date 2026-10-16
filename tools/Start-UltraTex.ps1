param(
    [ValidateSet('Legacy','Control','Hybrid','FullContextTest')]
    [string]$Mode = 'Legacy',
    [string]$ComfyRoot = '',
    [string]$PythonPath = '',
    [int]$StartupCapMiB = 6144,
    [string[]]$ExtraArgs = @(),
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if (-not $ComfyRoot) { $ComfyRoot = Join-Path $PSScriptRoot '../../..' }
$ComfyRoot = (Resolve-Path -LiteralPath $ComfyRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $ComfyRoot 'main.py'))) { throw 'ComfyRoot must contain main.py.' }
if (-not $PythonPath) { $PythonPath = Join-Path $ComfyRoot '../python_embeded/python.exe' }
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
if ($StartupCapMiB -le 0) { throw 'StartupCapMiB must be positive for explicit TEST/control modes.' }
$settings = @{
    UTX_ARCH_RESEARCH = $(if ($Mode -in @('Hybrid','FullContextTest')) { '1' } else { '0' })
    UTX_ARCH_GGUF_STORAGE = $(if ($Mode -eq 'Hybrid') { 'hybrid' } else { 'native' })
    UTX_ARCH_POLYPHASE = '0'
    UTX_FULL_CONTEXT_VAE = $(if ($Mode -eq 'FullContextTest') { '1' } else { '0' })
    UTX_FULL_CONTEXT_STARTUP_CAP_MB = $(if ($Mode -eq 'Legacy') { $null } else { "$StartupCapMiB" })
}
$arguments = @('-s',(Join-Path $ComfyRoot 'main.py'),'--enable-dynamic-vram','--async-offload','2','--use-pytorch-cross-attention') + $ExtraArgs
if ($Mode -eq 'FullContextTest') {
    Write-Host 'UltraTex FULL-CONTEXT TEST: changes normalization/attention context; quality equivalence to tiled output is unproven.'
    Write-Host 'Tested on RTX 5050 Laptop 8GB only. Read FULL_CONTEXT_TEST.md before opting in.'
    Write-Host 'This preset uses native GGUF storage. One complete scene passed sampled memory parity; repeated quality/performance acceptance remains pending.'
}
if ($Mode -eq 'Hybrid') { Write-Host 'Historical GGUF research setting: a combined test exceeded native dedicated memory by 64 MiB.' }
if ($DryRun) {
    [pscustomobject]@{ Python=$PythonPath; Arguments=$arguments; Mode=$Mode; Environment=$settings } | ConvertTo-Json -Depth 4
    return
}
$previous = @{}
foreach ($key in $settings.Keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$exitCode = 1
try {
    foreach ($key in $settings.Keys) { [Environment]::SetEnvironmentVariable($key,$settings[$key],'Process') }
    Push-Location -LiteralPath $ComfyRoot
    try { & $PythonPath @arguments; $exitCode = $LASTEXITCODE }
    finally { Pop-Location }
}
finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
exit $exitCode
