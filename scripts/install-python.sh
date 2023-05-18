#!/bin/bash
# This script is supposed to be run in the Docker image of the project

set -ex

VERSION="${1:-3.10.10}"

# Install Python
export DEBIAN_FRONTEND=noninteractive
sudo apt upgrade -y
sudo apt-get install -y wget build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev  libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev liblzma-dev
sudo rm -rf /var/lib/apt/lists/*
cd /tmp
sudo wget "https://www.python.org/ftp/python/$VERSION/Python-$VERSION.tgz"
sudo tar xzf Python-$VERSION.tgz
cd Python-$VERSION
sudo ./configure --enable-optimizations
