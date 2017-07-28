#!/bin/sh

# compile dependency in /opt/traildb/traildb

mkdir -p /opt/traildb
cd /opt/traildb

# shallow-ish copy of master branch of traildb/traildb
git clone --depth=50 https://github.com/traildb/traildb

# build traildb so
cd /opt/traildb/traildb
./waf configure
# actually needs root permissions to install into /usr/local
sudo ./waf install
