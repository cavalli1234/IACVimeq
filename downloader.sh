#!/usr/bin/env bash
wget http://http://46.101.188.29/ff_L10_B64.h5
wget http://http://46.101.188.29/plain_cnn_L10_cifar.h5
wget http://http://46.101.188.29/plain_cnn_L10_fivek.h5
wget http://http://46.101.188.29/u_net_best.h5
mv *.h5 resources/models/

