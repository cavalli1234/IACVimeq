# IACVimeq

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Install [Anaconda3](https://www.anaconda.com/distribution/#linux) with python 3.6
Remember to setup correctly the path, make sure to add the following line in your .bashrc (.zshrc) file substituiting the ${HOME} with the anaconda installation folder:

```console
export PATH="${HOME}/anaconda3/bin:$PATH"
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

## Running the tests

Once the environment is set up you can test the performance of each model and have samples of their output.

The following commands run the tests on 20 validation images. Change the -v option to run on more or fewer samples.

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
$ python source/run/fivek_test.py -v 20 -m plain -i plain_cnn_L10_fivek -b 128 -l 10
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

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
