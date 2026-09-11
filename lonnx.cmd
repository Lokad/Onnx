@echo off
pushd
@setlocal
set ERROR_CODE=0

REM From Alec Mev https://superuser.com/questions/35698/how-to-supress-terminate-batch-job-y-n-confirmation/715798#715798
IF [%JUSTTERMINATE%] == [OKAY] (
    SET JUSTTERMINATE=
    src\Lokad.Onnx.CLI\bin\Release\net10.0\Lokad.Onnx.CLI.exe %*
    CALL SET ERROR_CODE=%%ERRORLEVEL%%
) ELSE (
    SET JUSTTERMINATE=OKAY
    CALL %0 %* <NUL
    CALL SET ERROR_CODE=%%ERRORLEVEL%%
)

:end
@endlocal & popd & exit /B %ERROR_CODE%
