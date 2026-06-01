@echo off
REM ============================================================
REM Запуск теплового и динамического расчёта ДВС (Windows)
REM ============================================================
echo ========================================
echo  Тепловой и динамический расчёт ДВС
echo ========================================

REM 1. Проверка Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ОШИБКА: Python не найден.
    echo Установите Python 3.8+ с python.org и повторите попытку.
    pause
    exit /b 1
)

REM 2. Виртуальное окружение
if not exist ".venv" (
    echo ^>^>^> Создание виртуального окружения...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

REM 3. Установка зависимостей
echo ^>^>^> Установка зависимостей...
pip install -q -r requirements.txt

REM 4. Запуск
echo ^>^>^> Запуск расчёта...
python main.py

echo.
echo Готово! Результаты:
echo   - svg/               векторные диаграммы
echo   - dxf/               чертежи (DWG/DXF)
pause
