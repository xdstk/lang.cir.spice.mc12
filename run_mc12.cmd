@echo off
rem Micro-Cap 12, 64 bit (checked 25.10.2026); the 32-bit mc12.exe was not tested in batch mode
set MC="C:\MC12\Micro-Cap 12.exe"
%MC% /quiet @"%~dp0mc_list.bat"
echo Done. Output: *.ANO / *.TNO next to the .ckt files, log: mc_list.DOC
pause
