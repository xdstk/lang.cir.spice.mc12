@echo off
rem Fix the Micro-Cap path for your installation
set MC="C:\MC12\mc12_64.exe"
%MC% @"%~dp0mc_list.bat"
echo Done. Look for *.TNO / *.ANO next to the .ckt files
pause
