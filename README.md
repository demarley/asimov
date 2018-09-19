# Asimov

Minimalistic python framework for training HEP-based neural networks.  

Developed to read ROOT files (via [uproot](https://github.com/scikit-hep/uproot)) 
and pass the data to [keras](https://keras.io/) (+[tensorflow](https://www.tensorflow.org/)) for machine learning.

The framework uses a single text file to quickly prototype different hyperparameter configurations.
Relevant plots (to our analysis work) are generated with [hepPlotter](https://github.com/demarley/hepPlotter).
Hopefully this framework is useful for getting started and requires little modification for other users.

## Getting Started

Clone the repository and the hepPlotter repository (Asimov uses hepPlotter to make plots):

```
git clone https://github.com/demarley/asimov.git
git clone https://github.com/demarley/hepPlotter.git
```

## Software Versions

This software has been developed for a custom computing environment.
The Anaconda python installation is used to manage python libraries.

Module | Version
------ | -------
conda      | 4.4.10
matplotlib | 2.2.2
numpy      | 1.14.1
keras      | 2.0.8 (with Tensorflow backend)
tensorflow | 1.4.1 (tensorflow-gpu)
uproot     | 2.9.0
hepPlotter | v0.3 (developed using [Goldilocks](https://github.com/demarley/goldilocks))
cuda       | V9.0.176

Furthermore, this setup has access to an NVIDIA 1080Ti (with `NVIDIA-SMI 390.87`).


# Questions or Comments

Please submit an issue or PR.
