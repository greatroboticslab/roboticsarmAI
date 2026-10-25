@echo off
REM Usage:  run_stats.bat [path\to\dataset]
REM Default dataset path assumes this folder sits at the repo root next to dataset\: ..\dataset
set DS=%~1
if "%DS%"=="" set DS=..\dataset
python dataset_stats.py --dataset-dir "%DS%" --out dataset_stats_out
echo.
echo Open dataset_stats_out\dataset_stats_report.txt for the full report.
pause
