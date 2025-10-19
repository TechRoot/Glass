<#
Módulo de aritmética modular y teorema chino del resto.

Este módulo proporciona funciones para calcular el inverso modular de un
entero mediante el algoritmo de Euclides extendido y para resolver sistemas
de congruencias mediante el teorema chino del resto (CRT). Los nombres de
función siguen la convención PowerShell (`Get-InverseMod`, `Solve-CRT`).

Las funciones realizan validaciones básicas y generan errores en caso de
que no exista solución o inverso.
#>

function Get-GcdExtended {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][int]$A,
        [Parameter(Mandatory=$true)][int]$B
    )
    # Devuelve el máximo común divisor y los coeficientes de Bézout (g, x, y)
    if ($B -eq 0) {
        return @($A, 1, 0)
    }
    else {
        $res = Get-GcdExtended -A $B -B ($A % $B)
        $g = $res[0]; $x1 = $res[1]; $y1 = $res[2]
        $x = $y1
        $y = $x1 - [math]::Floor($A / $B) * $y1
        return @($g, $x, $y)
    }
}

function Get-InverseMod {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][int]$A,
        [Parameter(Mandatory=$true)][int]$M
    )
    # Calcula el inverso modular de A módulo M utilizando Euclides extendido.
    $res = Get-GcdExtended -A ([math]::Abs($A)) -B ([math]::Abs($M))
    $g = $res[0]; $x = $res[1]
    if ($g -ne 1) {
        throw "No existe inverso modular para $A módulo $M"
    }
    # Ajustar signo y módulo
    $inv = (($x % $M) + $M) % $M
    return $inv
}

function Solve-CRT {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory=$true)][int[]]$Rests,
        [Parameter(Mandatory=$true)][int[]]$Mods
    )
    # Resuelve un sistema de congruencias x ≡ Rests[i] (mod Mods[i])
    if ($Rests.Count -ne $Mods.Count) {
        throw "El número de restos y módulos debe ser igual"
    }
    # Inicializar x ≡ 0 (mod 1)
    $x = 0
    $M_total = 1
    for ($i = 0; $i -lt $Mods.Count; $i++) {
        $r = $Rests[$i]
        $m = $Mods[$i]
        # Calcular coeficientes de Bézout de M_total y m
        $res = Get-GcdExtended -A $M_total -B $m
        $g = $res[0]; $s = $res[1]; $t = $res[2]
        # Comprobar compatibilidad
        if ((($r - $x) % $g) -ne 0) {
            throw "Las congruencias dadas no son compatibles"
        }
        # Calcular el mínimo común múltiplo
        $lcm = [math]::Abs($M_total * $m) / $g
        # Ajustar la solución
        $factor = [math]::Floor((($r - $x) / $g) * $s)
        $x = $x + $M_total * $factor
        $x = (($x % $lcm) + $lcm) % $lcm
        $M_total = $lcm
    }
    return @{ Solution = $x; Modulus = $M_total }
}

Export-ModuleMember -Function Get-InverseMod, Solve-CRT