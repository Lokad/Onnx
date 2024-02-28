@echo off
@setlocal
set ERROR_CODE=0

echo Building Lokad Onnx projects..

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

echo Building Lokad Onnx Python interop project..
cd src\Lokad.Onnx.Interop
dotnet publish Lokad.Onnx.Interop.csproj -f net6.0 -p:PublishProfile=FolderProfile
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

