@echo off
@setlocal
set ERROR_CODE=0

dotnet restore src\Lokad.Onnx.sln
if not %ERRORLEVEL%==0  (
    echo Error restoring NuGet packages for Lokad.Onnx.sln.
    set ERROR_CODE=1
    goto End
)

dotnet build src\Lokad.Onnx.CLI\Lokad.Onnx.CLI.csproj /p:Configuration=Debug

if not %ERRORLEVEL%==0  (
    echo Error building Lokad.ONNX projects.
    set ERROR_CODE=2
    goto End
)

:End
@endlocal
exit /B %ERROR_CODE%

