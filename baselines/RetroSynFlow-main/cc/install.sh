#!/bin/bash

ENV_PATH="venv/retro"
REPO_PATH="flow_retro"

echo "Setting up Compute Canada environment for the project"
echo ""
echo "Will install the python environment in ~/$ENV_PATH"
echo "Expects the repository to be in ~/$REPO_PATH"
echo ""

echo "Loading modules"
echo "-------------------------------------------------------------------------"
source ~/$REPO_PATH/cc/load_modules.sh
echo ""

echo "Creating virtual environment ~/$ENV_PATH"
echo "-------------------------------------------------------------------------"
virtualenv --no-download ~/$ENV_PATH
echo ""

echo "Loading virtual environment"
echo "-------------------------------------------------------------------------"
source ~/$ENV_PATH/bin/activate
echo ""

echo "Installing python wheels from CC                (this might take a while)"
echo "-------------------------------------------------------------------------"
pip install -r ~/$REPO_PATH/requirements/pytorch.txt --no-index
echo ""
echo "Installing python wheels from PyPI              (this might take a while)"
echo "-------------------------------------------------------------------------"
pip install -r ~/$REPO_PATH/requirements/main.txt --no-index
echo ""

echo "Installing retflow"
echo "-------------------------------------------------------------------------"
pip install -e ~/$REPO_PATH
echo ""

echo "Installation finished"
echo "-------------------------------------------------------------------------"
echo ""
echo "To automatically load the environment on logging, "
echo "follow the following steps."
echo "    1. Create the environment file ~/$REPO_PATH/env-computecanada.sh" with the environment 
echo "       variables in the README.md file."
echo ""
echo ""
echo "    2. Add the following line to your ~/.bashrc"
echo ""
echo "       echo 'source ~/$REPO_PATH/cc/load_modules.sh         # Load python and cuda ' >> ~/.bashrc"
echo "       echo 'source ~/$ENV_PATH/bin/activate              # Load python env      ' >> ~/.bashrc"
echo "       echo 'source ~/$REPO_PATH/env-computecanada.sh       # Load env variables   ' >> ~/.bashrc"
echo ""
