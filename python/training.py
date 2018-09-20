"""
Created:        11 November  2016
Last Updated:    6 September 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Simple implementation of training with Keras
"""
import json
import util
import datetime

from foundation import Foundation

import numpy as np

import keras
from keras.models import Sequential,model_from_json,load_model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc



class Training(Foundation):
    """Training with Keras"""
    def __init__(self):
        super(Training,self).__init__()

        self.loss    = 'categorical_crossentropy'
        self.init    = 'normal'
        self.nNodes  = []
        self.metrics = ['accuracy']
        self.epochs     = 10
        self.optimizer  = 'adam'
        self.input_dim  = 1                  # number of features
        self.output_dim = 2                  # number of output dimensions
        self.batch_size = 32
        self.activations   = ['elu']         # https://keras.io/activations/
        self.nHiddenLayers = 1
        self.earlystopping = {}
        self.callbacks = []
        self.X_train   = None
        self.X_test    = None
        self.Y_train   = None
        self.Y_test    = None
        self.validation_split = 0.25
        self.test_split = 0.3



    def initialize(self):
        """Initialize a few parameters after they've been set by user"""
        super(Training,self).initialize()

        self.num_classes = len(self.classCollection.names())
        if self.earlystopping:
            self.callbacks = [EarlyStopping(**self.earlystopping)]


        ## -- Adjust model architecture parameters (flexibilty in config file)
        if len(self.nNodes)==1 and self.nHiddenLayers>0:
            # All layers (initial & hidden) have the same number of nodes
            self.msg_svc.DEBUG("DL : Setting all layers ({0}) to have the same number of nodes ({1})".format(self.nHiddenLayers+1,self.nNodes))
            nodes_per_layer = self.nNodes[0]
            self.nNodes = [nodes_per_layer for _ in range(self.nHiddenLayers+1)] # 1st layer + nHiddenLayers


        ## -- Adjust activation function parameter (flexibilty in config file)
        if len(self.activations)==1:
            # Assume the same activation function for all layers (input,hidden,output)
            self.msg_svc.DEBUG("DL : Setting input, hidden, and output layers ({0}) \n".format(self.nHiddenLayers+2)+\
                               "     to have the same activation function {0}".format(self.activations[0]) )
            activation = self.activations[0]
            self.activations = [activation for _ in range(self.nHiddenLayers+2)] # 1st layer + nHiddenLayers + output
        elif len(self.activations)==2 and self.nHiddenLayers>0:
            # Assume the last activation is for the output and the first+hidden layers have the first activation
            self.msg_svc.DEBUG("DL : Setting input and hidden layers ({0}) to the same activation function, {1},\n".format(self.nHiddenLayers+1,self.activations[0])+\
                               "     and the output activation to {0}".format(self.activations[1]) )
            first_hidden_act = self.activations[0]
            output_act       = self.activations[1]
            self.activations = [first_hidden_act for _ in range(self.nHiddenLayers+1)]+[output_act]

        return



    def train(self,**kwargs):
        """
        Train NN model

        @param kwargs   Pass extra arguments to limit plots (few supported options atm)
        """
        self.diagnostics(pre=self.runDiagnostics, **kwargs)

        self.build_model()
        self.train_model()
        self.save()
        self.evaluate_model()

        self.diagnostics(post=self.runDiagnostics)

        return


    ## Specific functions to perform training/inference tasks
    def build_model(self):
        """Construct the NN model -- only Keras support for now"""
        self.msg_svc.DEBUG("TRAINING : Build the neural network ")

        ## Declare the model
        self.model = Sequential()

        ## Add 1st layer
        self.model.add( Dense( int(self.nNodes[0]), input_dim=self.input_dim, kernel_initializer=self.init, activation=self.activations[0]) )

        ## Add hidden layer(s)
        for h in range(self.nHiddenLayers):
            self.model.add( Dense( int(self.nNodes[h+1]), kernel_initializer=self.init, activation=self.activations[h+1]) )

        ## Add the output layer
        self.model.add( Dense(self.output_dim,kernel_initializer=self.init, activation=self.activations[-1]) )

        ## Build the model
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        return


    def split_dataset(self):
        """Split dataframe into training/testing subsets"""
        tts = train_test_split(self.df[self.features].values,\
                               self.df['target'].values, \
                               test_size=self.test_split)
        self.X_train,self.X_test,self.Y_train,self.Y_test = tts

        return


    def train_model(self):
        """Setup for training the model using k-fold cross-validation"""
        self.msg_svc.DEBUG("TRAINING : Train the model!")
        self.split_dataset()

        # - Adjust shape of true values (matrix for multiple outputs)
        #   Only necessary for multi-classification, not binary
        if self.num_classes>2:
            self.Y_train = to_categorical(self.Y_train, num_classes=self.num_classes)
            self.Y_test  = to_categorical(self.Y_test,  num_classes=self.num_classes)

        ## Fit the model to training data & save the history (with validation!)
        self.history = self.model.fit(self.X_train,self.Y_train,\
                                      epochs=self.epochs,\
                                      validation_split=self.validation_split,\
                                      callbacks=self.callbacks,\
                                      batch_size=self.batch_size,verbose=self.verbose)

        return



    def evaluate_model(self):
        """Evaluate the model."""
        self.msg_svc.DEBUG("TRAINING : Evaluate the model ")

        train_predictions = self.predict(self.X_train) # predictions from training sample
        test_predictions  = self.predict(self.X_test)  # predictions from testing sample

        # -- store test/train prediction
        #    Need predictions for each class for each sample 
        #    (e.g., for top sample, what is the qcd prediction? For qcd sample, what is the qcd prediction? etc.)
        target_names = self.classCollection.names()
        h_tests  = {}
        h_trains = {}
        binning  = [0.1*i for i in range(11)]
        binary   = self.num_classes<=2

        # Predictions and ROC curves for each sample
        self.fpr = {}
        self.tpr = {}
        self.roc_auc = {}

        if binary:
            # binary classification
            # Make ROC curve from test sample
            fpr,tpr,_ = roc_curve( self.Y_test, test_predictions )
            self.fpr['binary'] = fpr
            self.tpr['binary'] = tpr
            self.roc_auc['binary'] = auc(fpr,tpr)

            # fill histograms of predictions for different classes for this sample
            for i,c in enumerate(self.classCollection):
                test_preds  = test_predictions[np.where(self.Y_test==c.value)]
                train_preds = train_predictions[np.where(self.Y_train==c.value)]

                h_tests[c.name]  = np.histogram(test_preds, bins=binning)
                h_trains[c.name] = np.histogram(train_preds,bins=binning)
        else:
            # multi-classification
            h_tests  = dict( (n,{}) for n in target_names )
            h_trains = dict( (n,{}) for n in target_names )

            for i,c in enumerate(self.classCollection):
                # Make ROC curve from test sample
                fpr,tpr,_ = roc_curve( self.Y_test[:,c.value], test_predictions[:,c.value] )
                self.fpr[c.name] = fpr
                self.tpr[c.name] = tpr
                self.roc_auc[c.name] = auc(fpr,tpr)

                # fill histograms of predictions for different classes for this sample
                category = np.array([0. for _ in range(len(target_names))])
                category[c.value] = 1.

                # array for each class prediction in single sample
                test_preds  = test_predictions[np.where(np.prod(self.Y_test==category, axis=-1))]
                train_preds = train_predictions[np.where(np.prod(self.Y_train==category, axis=-1))]

                for m in self.classCollection:
                    h_tests[c.name][m.name]  = np.histogram(test_preds[:,m.value], bins=binning)
                    h_trains[c.name][m.name] = np.histogram(train_preds[:,m.value],bins=binning)


        # Plot the predictions to compare test/train
        self.msg_svc.INFO("TRAINING : Plot the train/test predictions")
        self.plotter.prediction(h_trains,h_tests,binary)   # compare DNN prediction for different targets

        self.msg_svc.DEBUG("TRAINING :   Finished fitting model ")

        return



    def save(self):
        """Save the model; save the features to a json file to load via lwtnn later"""
        self.save_model()

        text = """  {
    "inputs": ["""

        for fe,feature in enumerate(self.features):
            comma = "," if fe!=len(self.features) else ""
            tmp = """
      {"name": "%(feature)s",
       "scale":  %(scale)d,
       "offset": %(offset)d}%(comma)s""" % {'feature':feature,'comma':comma,
                                            'scale':self.scale.get(feature,1),
                                            'offset':self.offset.get(feature,0)}
            text += tmp
        text += "],"
        text += """
    "class_labels": ["%(name)s"],
    "keras_version": "%(version)s",
    "miscellaneous": {}
  }
""" % {'version':keras.__version__,'name':self.dnn_name}

        varsFileName = self.output_dir+'/variables.json'
        varsFile     = open(varsFileName,'w')
        varsFile.write(text)

        return


## THE END ##
