#!/bin/bash
# This script is supposed to be run in the Docker image of the project

set -ex

# Add local user: either use the LOCAL_USER_ID if passed in at runtime or fallback
# export $(grep -v '^#' .env | xargs)
DEFAULT_USER=$(whoami)
DEFAULT_ID=$(id -u)
echo "DEFAULT_USER=${DEFAULT_USER}"
USER="${LOCAL_USER:${DEFAULT_USER}}"
USER_ID="${LOCAL_USER_ID:${DEFAULT_ID}}"

echo "UID: $USER_ID"
umask 022
VENV=/rl-lightning
ACTIVATE="source $VENV/bin/activate"
INSTALL_PROJECT="$ACTIVATE && poetry install; $ACTIVATE && python -m pip install -e ."

# Check who we are and based on that decide what to do
if [[ $USER = "root" ]] || [[ $USER = "" ]] || [[ -z $USER ]]; then
    # If root, just install brainiac-2
    bash -c "$INSTALL_PROJECT"
else
    # If not root, create user and give them root powers
    useradd --shell /bin/bash -u $USER_ID -o -c "" -m $USER
    export HOME=/home/$USER
    echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers
    echo "$USER ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USER
    sudo -H -u $USER bash -c 'echo "Running as USER=$USER, with UID=$UID"'
    sudo -H -u $USER bash -c "echo \"$ACTIVATE\" >> $HOME/.bashrc"
    sudo -H -u $USER bash -c "$INSTALL_PROJECT"
    exec gosu $USER "$@"
fi

