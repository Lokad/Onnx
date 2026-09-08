@echo off
setlocal EnableDelayedExpansion
set ERROR_CODE=0

echo Building Lokad Onnx package..

dotnet restore --tl:off -v minimal Lokad.Onnx.slnx
if not %ERRORLEVEL%==0 (
    echo Error restoring NuGet packages for Lokad.Onnx.slnx.
    set ERROR_CODE=1
    goto End
)

dotnet pack --tl:off --nologo -v minimal src\Lokad.Onnx\Lokad.Onnx.csproj -c Release
if not %ERRORLEVEL%==0 (
    echo Error packing Lokad.ONNX package.
    set ERROR_CODE=2
    goto End
)

:End
exit /B !ERROR_CODE!
