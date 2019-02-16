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

Explain how to run the automated tests for this system

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.
