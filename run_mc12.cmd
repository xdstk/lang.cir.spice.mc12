@echo off
rem Micro-Cap 12, 64 bit; the 32-bit C:\MC12\mc12.exe works the same way (both checked 25.10.2026)
set MC="C:\MC12\Micro-Cap 12.exe"
%MC% /quiet @"%~dp0mc_list.bat"
echo Done. Output: *.ANO / *.TNO next to the .ckt files, log: mc_list.DOC
pause
