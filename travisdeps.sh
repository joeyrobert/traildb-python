#!/bin/sh

# Get up to date libjudy
sudo apt-get install -y libarchive-dev pkg-config build-essential
wget https://mirrors.kernel.org/ubuntu/pool/universe/j/judy/libjudy-dev_1.0.5-5_amd64.deb \
     https://mirrors.kernel.org/ubuntu/pool/universe/j/judy/libjudydebian1_1.0.5-5_amd64.deb
sudo dpkg -i libjudy-dev_1.0.5-5_amd64.deb libjudydebian1_1.0.5-5_amd64.deb

# compile dependency in /opt/traildb/traildb

mkdir -p /opt/traildb
cd /opt/traildb

# shallow-ish copy of master branch of traildb/traildb
git clone --depth=50 https://github.com/traildb/traildb

# build traildb so
cd /opt/traildb/traildb
sudo ./waf configure
# actually needs root permissions to install into /usr/local
sudo ./waf install