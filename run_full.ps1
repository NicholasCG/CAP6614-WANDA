param(
    [string]$Model = "meta-llama/Llama-2-7b-hf",
    [int]$SequenceLength = 2048,
    [int]$NSamples = 128,
    [string]$DType = "float16",
    [string]$ResultsRoot = ".\results\full_unstructured",
    [int]$LayerSensitivityAt = 50
)

$ErrorActionPreference = "Stop"

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Script
    )

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "[RUN] $Name" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    & $Script
}

function Ensure-Dir {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

$ProjectRoot = Get-Location
$env:PYTHONPATH = $ProjectRoot.Path

$sparsities = @(0.4, 0.5, 0.6, 0.7, 0.8)

Ensure-Dir $ResultsRoot

@{
    model = $Model
    sequence_length = $SequenceLength
    nsamples = $NSamples
    dtype = $DType
    sparsities = $sparsities
    layer_sensitivity_at_percent = $LayerSensitivityAt
    started = (Get-Date).ToString("s")
} | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $ResultsRoot "run_config.json")

foreach ($Sparsity in $sparsities) {
    $Tag = [int]([Math]::Round($Sparsity * 100))
    $RunDir = Join-Path $ResultsRoot ("s{0}" -f $Tag)

    Ensure-Dir $RunDir
    Ensure-Dir (Join-Path $RunDir "magnitude")
    Ensure-Dir (Join-Path $RunDir "wanda")
    Ensure-Dir (Join-Path $RunDir "sparsegpt")
    Ensure-Dir (Join-Path $RunDir "layer_sensitivity")

    Run-Step "Magnitude pruning @ $Tag% sparsity" {
        python .\main.py `
            --model $Model `
            --prune_method magnitude `
            --sparsity $Sparsity `
            --sequence_length $SequenceLength `
            --nsamples $NSamples `
            --dtype $DType `
            --output_dir (Join-Path $RunDir "magnitude")
    }

    Run-Step "Wanda pruning @ $Tag% sparsity" {
        python .\main.py `
            --model $Model `
            --prune_method wanda `
            --sparsity $Sparsity `
            --sequence_length $SequenceLength `
            --nsamples $NSamples `
            --dtype $DType `
            --output_dir (Join-Path $RunDir "wanda")
    }

    Run-Step "SparseGPT pruning @ $Tag% sparsity" {
        python .\main.py `
            --model $Model `
            --prune_method sparsegpt `
            --sparsity $Sparsity `
            --sequence_length $SequenceLength `
            --nsamples $NSamples `
            --dtype $DType `
            --output_dir (Join-Path $RunDir "sparsegpt")
    }

    if ($Tag -eq $LayerSensitivityAt) {
        Run-Step "Layer sensitivity @ $Tag% sparsity" {
            python .\experiments\run_layer_sensitivity.py `
                --model $Model `
                --output_dir (Join-Path $RunDir "layer_sensitivity") `
                --sparsity $Sparsity `
                --sequence_length $SequenceLength `
                --nsamples $NSamples `
                --dtype $DType `
                --group_mode block
        }
    }
}

Write-Host ""
Write-Host "All experiments finished." -ForegroundColor Green
Write-Host "Results saved to $ResultsRoot" -ForegroundColor Green