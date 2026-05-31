# GSoC mae-lensing 環境一鍵安裝
# 用法: 在 PowerShell 中執行
#   cd C:\Users\USER\Downloads\GSoC
#   .\setup_env.ps1

Write-Host "=== Step 1: PyTorch with CUDA 12.4 (work with CUDA 12.7 driver) ===" -ForegroundColor Cyan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

Write-Host "`n=== Step 2: scientific packages ===" -ForegroundColor Cyan
pip install optuna gdown

Write-Host "`n=== Step 3: PennyLane (量子) ===" -ForegroundColor Cyan
pip install pennylane pennylane-lightning

Write-Host "`n=== Verification ===" -ForegroundColor Green
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python -c "import pennylane as qml; print('PennyLane:', qml.__version__)"
python -c "import optuna; print('Optuna:', optuna.__version__)"

Write-Host "`n=== Done. Next: download data with download_data.py ===" -ForegroundColor Green
