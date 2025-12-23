@echo off
@setlocal
set ERROR_CODE=0

echo Building Lokad Onnx package..

dotnet restore Lokad.Onnx.slnx
if not %ERRORLEVEL%==0  (
    echo Error restoring NuGet packages for Lokad.Onnx.slnx.
    set ERROR_CODE=1
    goto End
)

dotnet build src\Lokad.Onnx.Package\Lokad.Onnx.Package.csproj /p:Configuration=Release
if not %ERRORLEVEL%==0  (
    echo Error building Lokad.ONNX package.
    set ERROR_CODE=2
    goto End
)

:End
@endlocal
exit /B %ERROR_CODE%

