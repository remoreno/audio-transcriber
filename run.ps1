if (-not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    Write-Host "Installing requirements..."
    & ".\venv\Scripts\pip.exe" install -r requirements.txt
}

& ".\venv\Scripts\python.exe" "main.py"
Read-Host "Press Enter to close"