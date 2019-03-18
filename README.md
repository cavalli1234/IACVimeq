# IACVimeq

Human expert photo enhancement applies arbitrary color transformations on different semantic areas of the photo, making it an unstructured problem to be approached directly. Histogram equalization, on the other side, is a simple algorithm that can easily be analysed to suggest structural constraint on a deep architecture to improve learning of meaningful color transformations. In this paper we design deep architectures for color transformations specialized in histogram equalization and we show their generalization capabilities to the expert photo enhancement task. Experimental results suggest that one of the proposed architectures can achieve slightly better results with respect to traditional architectures using much less parameters.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install [Anaconda3](https://www.anaconda.com/distribution/#linux) with python 3.6
Remember to setup correctly the path, make sure to add the following line in your .bashrc (.zshrc) file substituiting the ${HOME} with the anaconda installation folder:

```console
export PATH="${HOME}/anaconda3/bin:$PATH"
```

At this point you should reset the console (close it and open again!)

You should have the program unzip installed in order to extract the fivek dataset, if not install it
```zsh
$ sudo apt install unzip
```

### Installing

For the installations run the following commands:

clone the repository
```zsh
$ git clone https://github.com/cavalli1234/IACVimeq
```

setup the environment
```zsh
$ cd IACVimeq
$ chmod +x setup_environment.sh
$ ./setup_environment.sh
```

download our models
```zsh
$ chmod +x downloader.sh
$ ./downloader.sh
```

if you want to do some tests on the MIT-fivek dataset you need to download it as well (8GB download)
```zsh
$ chmod +x fivek_downloader.sh
$ ./fivek_downloader.sh
```

The models and a part of the fivek dataset are already included in the repository

## Running the tests

Once the environment is set up you can test the performance of each model and have samples of their output.

The first plot of the test script shows the worst validation sample, followed by three random samples.
The shown images from left to right are: original, ground truth, network output, difference.

The second plot of the test script shows an histogram of the losses for each validation image evaluated.

The following commands run the tests on 20 validation images. Change the -v option to run on more or less samples.

Before running tests, run the following command:

```zsh
$ source activate IACVimeq
```

Pixel-wise Fully Connected on cifar10:
```zsh
$ python source/run/cifar_test.py -v 20 -m ff -i ff_L10_B64 -b 64 -c 1
```

Plain CNN on cifar10:
```zsh
$ python source/run/cifar_test.py -v 20 -m plain -i plain_cnn_L10_cifar -b 128 -l 10 -c 1 --from-fresh
```

Histogram CNN on cifar10:
```zsh
$ python source/run/cifar_test.py -v 20 -m hist -i hist_building_cnn_L10_B128_superGood -c 1
```

Plain CNN on FiveK:
```zsh
$ python source/run/fivek_test.py -v 20 -m plain -i plain_cnn_L10_fivek -b 128 -l 10 --from-fresh
```

U-net on FiveK:
```zsh
$ python source/run/fivek_test.py -v 20 -m unet -i u_net_best
```

Histogram CNN on FiveK:
```zsh
$ python source/run/fivek_test.py -v 20 -m hist -i hist_building_cnn_L5_B64_fivek
```


## Authors

* **Gianpaolo Di Pietro**
* **Luca Cavalli**
