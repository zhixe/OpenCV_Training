@echo off
:: Execute the PowerShell script with bypassing the execution policy
powershell -ExecutionPolicy Bypass -File "batch/00_installation.ps1"
exit