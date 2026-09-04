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

dotnet build --tl:off --nologo -v minimal src\Lokad.Onnx.Package\Lokad.Onnx.Package.csproj /p:Configuration=Release
if not %ERRORLEVEL%==0 (
    echo Error building Lokad.ONNX package.
    set ERROR_CODE=2
    goto End
)

:End
exit /B !ERROR_CODE!
