@echo off

REM This batch script will create a venv for you and install
REM all of the necessary packages that you will require to
REM download the dataset, train the model and then evaluate it.

REM NOTE: The script will leave you in the venv. If you want to
REM exit the venv, simply type `deactivate` in the terminal.

echo Creating venv and installing packages...
py -m venv venv
call venv\Scripts\activate
pip install --upgrade huggingface_hub
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm
echo Done!