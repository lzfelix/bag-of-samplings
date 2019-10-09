#/bin/bash

yes | rm -rf healthy/ patients/ Signal/ > /dev/null

curl -O http://www2.fc.unesp.br/~papa/pub/datasets/Handpd/NewHealthy/Signal.zip
unzip Signal.zip > /dev/null

echo 'Extracting healthy signals'
rm Signal.zip
rm -rf __MACOSX/
chmod -x Signal/*
mv Signal/ healthy/

curl -O http://www2.fc.unesp.br/~papa/pub/datasets/Handpd/NewPatients/Signal.zip
unzip Signal.zip > /dev/null

echo 'Extracting patients signals'
rm Signal.zip
rm -rf __MACOSX/
chmod -x Signal/*
mv Signal/ patients/

