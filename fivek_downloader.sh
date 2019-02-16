#!/usr/bin/env bash
echo "Downloading fivek dataset..."
wget http://46.101.188.29/fivek.zip
echo "Unzipping dataset..."
mkdir resources/dataset
mv fivek.zip resources/dataset/
unzip -oq resources/dataset/fivek.zip -d resources/dataset/
rm resources/dataset/fivek.zip
echo "Done!"
