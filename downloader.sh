#!/usr/bin/env bash
wget http://http://46.101.188.29/ff_L10_B64.h5
wget http://http://46.101.188.29/plain_cnn_L10_cifar.h5
wget http://http://46.101.188.29/hist_building_cnn_L10_B128_superGood.h5
wget http://http://46.101.188.29/plain_cnn_L10_fivek.h5
wget http://http://46.101.188.29/u_net_best.h5
wget http://http://46.101.188.29/hist_building_cnn_L5_B64_fivek.h5
mv *.h5 resources/models/

