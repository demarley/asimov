"""
Created:         6 September 2018
Last Updated:    6 September 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Base class for performing deep learning in HEP
using Keras with a Tensorflow backend

Designed for running with specific set of software
--> not guaranteed to work in CMSSW environment!

Modules used:
> UPROOT:     https://github.com/scikit-hep/uproot
> KERAS:      https://keras.io/
> TENSORFLOW: https://www.tensorflow.org/
> LWTNN:      https://github.com/lwtnn/lwtnn

NB PyTorch currently unsupported
"""
import json
import datetime
from collections import Counter

import util
from empire import Empire

import uproot
import numpy as np
import pandas as pd

# fix random seed for reproducibility
seed = 2018
np.random.seed(seed)


class Foundation(object):
    """Deep Learning base class"""
    def __init__(self):
        self.date = datetime.date.today().strftime('%d%b%Y')

        ## Handling NN objects and data -- set in the class
        self.df  = None             # dataframe containing physics information
        self.fpr = {}               # ROC curve: false positive rate
        self.tpr = {}               # ROC curve: true positive rate
        self.roc_auc = {}           # ROC curve: Area under the curve
        self.model   = None         # Keras model
        self.history = None         # model history
        self.plotter = None         # plotter module
        self.scale   = {}           # scale for normalizing input features
        self.offset  = {}           # offset for normalizing input features
        self.metadata = {}          # Metadata to store for accessing later, if necessary
        self.features = []          # List of features used in the analysis
        self.classes  = {}          # classes for neural network {"name":value}
        self.classCollection = None # storing information about the different classes
        self.sample_labels = {}
        self.variable_labels = {}

        ## NN architecture & parameters -- set by config file
        self.treename   = 'features'    # Name of TTree to access in ROOT file (via uproot)
        self.lwtnn      = True          # export (& load model from) files for LWTNN
        self.dnn_name   = "dnn"         # name to access in lwtnn ('variables.json')
        self.hep_data   = ""            # Name for loading features (physics data) -- assumes all data in one file
        self.model_name = ""            # Name for saving/loading model
        self.output_dir = 'data/dnn/'   # directory for storing NN data
        self.runDiagnostics = False     # Make plots pre/post training
        self.msg_svc = None
        self.verbose = True
        self.equal_statistics = True    # Equal statistics for each class in the df



    def initialize(self):
        """Initialize a few parameters after they've been set by user"""
        self.verbose = self.msg_svc.compare("WARNING")  # if verbose level < "WARNING", verbose output!

        # Use input filename to generate model name
        if not self.model_name:
            self.model_name = self.hep_data.split('/')[-1].split('.')[0]+'_'+self.date

        # Store NN classes in custom object
        self.classCollection = util.NNClassCollection()
        for n,v in self.classes.iteritems():
            nn_class = util.NNClass(n)
            nn_class.value = v
            self.classCollection.append(nn_class)


        ## -- Plotting framework
        self.plotter = Empire()  # class for plotting relevant NN information
        self.plotter.output_dir   = self.output_dir
        self.plotter.image_format = 'pdf'
        self.plotter.features     = self.features
        self.plotter.msg_svc      = self.msg_svc
        self.plotter.sample_labels   = self.sample_labels
        self.plotter.variable_labels = self.variable_labels
        self.plotter.initialize(self.classCollection)

        return



    def predict(self,data=None):
        """Return the prediction from a test sample"""
        self.msg_svc.DEBUG("DL : Get the DNN prediction")
        if data is None:
            self.msg_svc.ERROR("DL : predict() given NoneType data. Returning -999.")
            self.msg_svc.ERROR("DL : Please check your configuration!")
            return -999.
        return self.model.predict( data )



    def load_data(self,extra_variables=[]):
        """
        Load the physics data (flat ntuple) for NN using uproot
        Convert to DataFrame for easier slicing 

        @param extra_variables   If there are extra variables to plot/analyze, 
                                 that aren't features of the NN, include them here
        """
        self.msg_svc.INFO("DL : Load HEP data")

        file = uproot.open(self.hep_data)
        data = file[self.treename]
        self.df = data.pandas.df( self.features+extra_variables )

        self.metadata['metadata'] = file['metadata'] # names of samples, target values, etc.

        return


    def preprocess_data(self):
        """Manipulate dataframe and keep only the necessary data"""
        class_dfs = []
        min_size  = self.df.shape[0]
        for k in self.classCollection:
            tmp = self.df[ self.df.target==k.value ]
            class_dfs.append(tmp)
            if tmp.shape[0]<min_size: 
                min_size=tmp.shape[0]

        # Make the dataset sizes equal for the different classes
        if self.equal_statistics:
            for td,cdf in enumerate(class_dfs):
                # shuffle entries and select first events up to 'min_size'
                if cdf.shape[0]>min_size:
                    class_dfs[td] = class_dfs[td].sample(frac=1)[0:min_size]

        self.df = pd.concat( class_dfs ).sample(frac=1) # re-combine & shuffle entries

        self.df = self.df[ (self.df.AK4_deepCSVb>=0) & \
                           (self.df.AK4_deepCSVbb>=0) & \
                           (self.df.AK4_deepCSVc>=0) & \
                           (self.df.AK4_deepCSVl>=0) ]

        return


    def save_model(self):
        """Save the model for use later"""
        self.msg_svc.INFO("DL : Save model")

        output = self.output_dir+'/'+self.model_name

        if self.lwtnn:
            ## Save to format for LWTNN
            with open(output+'_model.json', 'w') as outfile:
                outfile.write(self.model.to_json())          # model architecture

            self.model.save_weights(output+'_weights.h5')    # model weights
        else:
            ## Save to h5 format
            self.model.save('{0}.h5'.format(output))


        ## Keep track of the different features used in this instance
        featureKeysFile = self.output_dir+'/features.json'
        try:
            featureKeys = json.load(open(featureKeysFile))
        except IOError:
            featureKeys = {}
        
        featureKey = -1
        for key in featureKeys.keys():
            if Counter(featureKeys[key])==Counter(config.features):
                featureKey = int(key)
                break
        if featureKey<0:
            keys = featureKeys.keys()
            featureKey = max([int(i) for i in keys])+1 if keys else 0
            featureKeys[str(featureKey)] = self.features
            with open(featureKeysFile,'w') as outfile:
                json.dump(featureKeys,outfile)


        ## Record different hyperparameters used in this instance
        NN_parameters = ['epochs','batch_size','loss','optimizer','metrics',
                         'activations','nHiddenLayers','nNodes','input_dim']
        outputFile = open(self.output_dir+'/ABOUT.txt','w')
        outputFile.write(" * NN Setup * \n")
        outputFile.write(" ------------ \n")
        outputFile.write(" NN parameters: \n")

        for NN_parameter in NN_parameters:
            outputFile.write( NN_parameter+": "+str(getattr(self,NN_parameter))+"\n" )
        outputFile.write( "\n NN Features: \n" )
        for feature in self.features:
            outputFile.write("  >> "+feature+"\n" )
        outputFile.close()

        return


    def load_model(self):
        """Load existing model to make plots or predictions"""
        self.model = None

        if self.lwtnn:
            model_json = open(self.model_name+"_model.json",'r').read()
            self.model = model_from_json(model_json)
            self.model.load_weights(self.model_name+"_weights.h5")
            self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=self.metrics)
        else:
            self.model = load_model('{0}.h5'.format(self.model_name))

        return



    def diagnostics(self,pre=False,post=False):
        """Diagnostic tests of the NN"""
        self.msg_svc.INFO("DL : Diagnostics")

        # Plots to make pre-training
        if pre:
            self.msg_svc.INFO("DL : -- pre-training :: features")
            self.plotter.feature(self.df)           # compare features

            self.msg_svc.INFO("DL : -- pre-training :: correlations")
            corrmats = {}
            for c in self.classCollection:
                t_ = self.df[self.df.target==c.value].drop('target',axis=1)
                corrmats[c.name] = t_.corr()
            self.plotter.correlation(corrmats)       # correlations between features

            self.msg_svc.INFO("DL : -- pre-training :: separations")
            self.plotter.separation()                # separations between classes

        # Plots to make post-training/testing
        if post:
            self.msg_svc.INFO("DL : -- post-training :: ROC")
            self.plotter.ROC(self.fpr,self.tpr,self.roc_auc)   # Signal vs background eff
            self.msg_svc.INFO("DL : -- post-training :: History")
            self.plotter.history(self.history)                 # loss vs epoch

        return


## THE END ##
