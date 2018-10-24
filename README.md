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
cd hepPlotter/
git checkout tags/v0.4.2    # current compatibility
```

Please see the `examples/` directory for an example on using this framework with data from the Higgs Boson Machine Learning Challenge.

## Overview & Workflow

This (simple) framework serves as an interface between HEP data and machine learning libraries (keras).
The uproot package allows us to open ROOT data files in a python environment and port the data directly to a pandas dataframe.
The dataframe can then be passed to keras as needed to do the training.
Using hepPlotter, we can make plots of the features, correlations, etc. to visualize the ML performance.

The input root file is assumed to be flat and each 'event' in the TTree contains branches necessary for training 
(the branches in the TTree need to match the names of the features provided in the configuration file).
If you're designing an algorithm that discriminates objects, e.g., jets, then each 'event' in the TTree must represent each jet, rather than the actual physics event.   
Please inspect `example.root` in the `examples/` directory for more information.  
_NB: This is an area for future development, but the current simplicity of this setup prevents that.  Furthermore, the author uses their existing workflow (a C++ environment) to generate flat ntuples._

Files | Description
----- | -----------
`foundation.py` (`training.py` and `inference.py` inherit from this) | Base class
`empire.py` | Plotting class
`config.py` | Configuration class (reads text file and sets NN framework)
`util.py`   | Misc. utility functions


### Configuration

A single text file dictates the NN architecture, what data to process, where store outputs, and what features to use, among other things.
An example is provided here: `examples/example_config.txt`.
In this file, the list of features are comma-separated and they match the branches in `example.root` that we want to use for the training (noted above).
The class `python/config.py` reads the text file and stores the relevant data for use by the various NN classes.

### Selection
To apply further selection on dataframe, you can create a list of strings that will be parsed
and then used to select events from the dataframe.

```python
# slices = ['BRANCH <OP> VALUE',...]
# where : 'BRANCH' is the branch name in the root file
#       : <OP> is the mathematical operator, e.g., '>' or '<='
#       : 'VALUE' is the value the branch is being compared to
# e.g., for the examples directory:
slices = ['mass_MMC > 0']   # this would ensure you don't train on events with mass_MMC=-999.
dnn.preprocess_data(slices)
```

### Output

The model and figures are saved for further inspection and use in a c++ production environment.
The [LWTNN](https://github.com/lwtnn/lwtnn) framework is the default output option, but it is possible to save the model in other forms.

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

### Google Colab

NB: For those interested, it is possible to run this setup using Google Colab.
Feel free to have a look at [asimov_demo.ipynb](https://colab.research.google.com/drive/1sb0-Yy71aS1zPOqIJZw9Y-BwEN8SVV0l). 
This is modeled after the example notebook [0-simple.ipynb](https://github.com/demarley/asimov/blob/master/examples/0-simple.ipynb) that uses the data from the Higgs Boson ML Challenge.  
There are a few known issues (e.g., LaTeX isn't available for plot labels), but the code runs rather successfully on both TPUs and GPUs.

# Questions or Comments

Please submit an issue or PR.
