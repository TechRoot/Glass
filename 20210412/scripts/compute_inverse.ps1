<#!
CLI de PowerShell para aritmética modular y el teorema chino del resto.

Este script utiliza el módulo `modular_arithmetic.psm1` para calcular el
inverso modular de un entero o para resolver un sistema de congruencias
mediante el teorema chino del resto. Incluye soporte para `--dry-run`
(simulación) y `--confirm` para guardar resultados en trazas.

Uso:
  # Calcular inverso de A mod M
  ./compute_inverse.ps1 -Command inverse -A 3 -M 11

  # Resolver x ≡ a_i (mod m_i)
  ./compute_inverse.ps1 -Command crt -Rests 2,3,2 -Mods 3,5,7
#>

param(
    [Parameter(Mandatory=$true)][ValidateSet("inverse", "crt")]
    [string]$Command,
    [int]$A,
    [int]$M,
    [int[]]$Rests,
    [int[]]$Mods,
    [switch]$Confirm,
    [switch]$DryRun = $true,
    [string]$TracePath = "../data/traces/block3_trace.csv"
)

# Cargar el módulo de aritmética modular
$modulePath = Join-Path -Path $PSScriptRoot -ChildPath "..\lib\modular_arithmetic.psm1"
Import-Module $modulePath -ErrorAction Stop

function Write-Trace {
    param(
        [string]$action,
        [string]$input,
        [string]$result
    )
    $date = Get-Date -Format o
    $line = "$date,$action,$input,$result"
    $fullPath = Resolve-Path -Path (Join-Path -Path $PSScriptRoot -ChildPath $TracePath)
    if (-not (Test-Path $fullPath)) {
        # crear directorio si no existe
        $dir = Split-Path -Parent $fullPath
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
        "timestamp,action,input,result" | Out-File -FilePath $fullPath -Encoding utf8
    }
    $line | Out-File -FilePath $fullPath -Append -Encoding utf8
}

if ($Command -eq "inverse") {
    if ($A -eq $null -or $M -eq $null) {
        Write-Error "Debe especificar -A y -M para calcular el inverso"
        exit 1
    }
    try {
        $inv = Get-InverseMod -A $A -M $M
        Write-Output "Inverso de $A módulo $M = $inv"
        if ($Confirm) {
            Write-Trace -action "inverse" -input "A=$A;M=$M" -result $inv
        }
    }
    catch {
        Write-Error $_
    }
}
elseif ($Command -eq "crt") {
    if ($Rests -eq $null -or $Mods -eq $null) {
        Write-Error "Debe especificar -Rests y -Mods para CRT"
        exit 1
    }
    try {
        $res = Solve-CRT -Rests $Rests -Mods $Mods
        Write-Output "Solución CRT: x ≡ $($res.Solution) (mod $($res.Modulus))"
        if ($Confirm) {
            $input = "Rests=$($Rests -join ',');Mods=$($Mods -join ',')"
            $result = "x=$($res.Solution);M=$($res.Modulus)"
            Write-Trace -action "crt" -input $input -result $result
        }
    }
    catch {
        Write-Error $_
    }
}