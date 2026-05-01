$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host ""
Write-Host "森林损失情景控制与蒙特卡洛碳储量模拟系统 V1.0"
Write-Host "简称：森林损失碳模拟"
Write-Host "================================================"
Write-Host "这个窗口就是系统运行终端，请不要关闭。"
Write-Host ""

python run_system.py

Write-Host ""
Write-Host "系统已停止。"
Read-Host "按回车键退出"
