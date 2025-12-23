@echo off
@setlocal
set ERROR_CODE=0

echo Building Lokad Onnx projects..

dotnet restore --tl:off -v minimal Lokad.Onnx.slnx
if not %ERRORLEVEL%==0  (
    echo Error restoring NuGet packages for Lokad.Onnx.slnx.
    set ERROR_CODE=1
    goto End
)

dotnet build --tl:off --nologo -v minimal src\Lokad.Onnx.CLI\Lokad.Onnx.CLI.csproj /p:Configuration=Release
if not %ERRORLEVEL%==0  (
    echo Error building Lokad.ONNX projects.
    set ERROR_CODE=2
    goto End
)

echo Building Lokad Onnx Python interop project..
cd src\Lokad.Onnx.Interop
dotnet publish --tl:off --nologo -v minimal Lokad.Onnx.Interop.csproj -f net6.0 -p:PublishProfile=FolderProfile
if not %ERRORLEVEL%==0  (
    echo Error building Lokad.ONNX projects.
    set ERROR_CODE=2
    cd..\..
    goto End
)

cd ..\..\
echo Building Lokad Onnx projects complete.

:End
@endlocal
exit /B %ERROR_CODE%

