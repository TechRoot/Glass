<#
Pruebas Pester para el módulo de aritmética modular.

Comprueba la correcta implementación del inverso modular y del teorema chino
del resto. Ejecutar con `Invoke-Pester` desde PowerShell 7.
#>

Import-Module "$PSScriptRoot/../../lib/modular_arithmetic.psm1"

Describe "Pruebas de aritmética modular" {
    It "Encuentra el inverso de 3 mod 11" {
        (Get-InverseMod -A 3 -M 11) | Should -Be 4
    }
    It "Encuentra el inverso de 10 mod 17" {
        (Get-InverseMod -A 10 -M 17) | Should -Be 12
    }
    It "Lanza error si no existe inverso" {
        { Get-InverseMod -A 6 -M 9 } | Should -Throw
    }
    It "Resuelve un sistema de congruencias simple" {
        $res = Solve-CRT -Rests @(2,3,2) -Mods @(3,5,7)
        $res.Solution | Should -Be 23
        $res.Modulus | Should -Be 105
    }
}