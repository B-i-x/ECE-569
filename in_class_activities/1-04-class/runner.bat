@echo off
REM Load the CUDA environment (if required)
CALL nvcc -o matrixadd.exe ..\madd_template.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

REM Check if compilation was successful
IF %ERRORLEVEL% NEQ 0 (
    echo Compilation failed.
    GOTO :end
)

echo Compilation successful, running the program...
matrixadd.exe 6000
:end
pause