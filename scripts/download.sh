#!/bin/bash

wget -O data.zip https://at.ispras.ru/owncloud/index.php/s/BufPcq5HqKpC12o/download
wget -O saved_models.zip https://at.ispras.ru/owncloud/index.php/s/NhjGF9ZEUNzDLDy/download
unzip data.zip
unzip saved_models.zip
rm -rf data.zip saved_models.zip
