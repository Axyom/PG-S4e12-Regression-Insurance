@echo off
set COMP_NAME=playground-series-s4e12
set FILE_PATH=submission.csv
set MESSAGE=Clean XGBOOST local v1

kaggle competitions submit -c %COMP_NAME% -f %FILE_PATH% -m "%MESSAGE%"
pause

