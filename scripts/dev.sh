#!/bin/bash
set -ex

# Add local user: either use the LOCAL_USER_ID if passed in at runtime or fallback
# export $(grep -v '^#' .env | xargs)
USER="vscode"
USER_ID=$(id -u)

echo "Starting with UID: $USER_ID"
useradd --shell /bin/bash -u $USER_ID -o -c "" -m $USER
export HOME=/home/$USER

# sudo chown user -R /workdir /venv
sudo -H -u $USER bash -c 'echo "Running as USER=$USER, with UID=$UID"'
sudo -H -u $USER bash -c 'source /venv/bin/activate && python -m pip install -e .'
umask 022
source /venv/bin/activate
su - $USER