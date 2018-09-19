"""
Created:        16 August    2018
Last Updated:    6 September 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Configuration class for getting/setting parameters
to use in the NN.
"""
import os
import sys
import util



class Config(object):
    """Configuration object that handles the setup"""
    def __init__(self,file):
        self.filename = file

        self.set_defaults()                          # set default config options
        self.configuration = util.read_config(file)  # read the configuration
        self.setAttributes()                         # set attributes of class


    def get(self,param):
        """Return values of the configuration to the user"""
        value = None

        try:
            value = self.configuration[param]
        except KeyError:
            print "WARNING :: CONFIG : The configuration file does not contain {0}".format(param)
            print "WARNING :: CONFIG : Using default value."
            try:
                value = self.defaults[param]
            except KeyError:
                raise KeyError("There is no default value for {0}".format(param))

        return value



    def setAttributes(self):
        """Set attributes of class for the configurations"""
        setattr(self,'model_name',   self.get('model_name'))
        setattr(self,'framework',    self.get('framework') )
        setattr(self,'runTraining',  util.str2bool( self.get('runTraining') ))
        setattr(self,'runInference', util.str2bool( self.get('runInference') ))
        setattr(self,'hep_data',     self.get('hep_data'))
        setattr(self,'treename',     self.get('treename'))
        setattr(self,'dnn_data',     self.get('dnn_data'))
        setattr(self,'output_path',  self.get('output_path'))
        setattr(self,'nHiddenLayers',int( self.get('nHiddenLayers') ))
        setattr(self,'nNodes',       self.get('nNodes').split(','))
        setattr(self,'epochs',       int( self.get('epochs') ))
        setattr(self,'batch_size',   int( self.get('batch_size') ))
        setattr(self,'loss',         self.get('loss'))
        setattr(self,'optimizer',    self.get('optimizer'))
        setattr(self,'metrics',      self.get('metrics').split(','))
        setattr(self,'kfold_splits', int( self.get('kfold_splits') ))
        setattr(self,'init',         self.get('init'))
        setattr(self,'output_dim',   int( self.get('output_dim') ))
        setattr(self,'features',     self.get('features').split(','))
        setattr(self,'percentile',   int( self.get('percentile') ))
        setattr(self,'activation',   self.get('activation') )
        setattr(self,'nEntries',     int( self.get('nEntries') ))
        setattr(self,'verbose_level',self.get('verbose') )

        return


    def set_defaults(self):
        """Set default values for configurations"""
        self.defaults = {'framework':'keras',
                         'runTraining':False,
                         'runInference': False,
                         'hep_data':None,
                         'treename':"",
                         'dnn_data':None,
                         'output_path':'./',
                         'nHiddenLayers':1,
                         'nNodes':5,
                         'epochs':10,
                         'batch_size':32,
                         'loss':'binary_crossentropy',
                         'optimizer':'adam',
                         'metrics':['accuracy'],
                         'init':'normal',
                         'output_dim':1,
                         'features':[],
                         'activation':'elu',
                         'nEntries':-1,
                         'model_name':'',
                         'verbose_level':'INFO'}

        return


    def __str__(self):
        """Specialized print statement for this class"""
        command = " Asimov : Neural Network Configuration \n"

        keys = [i for i in self.__dict__.keys() if not i.startswith('_')]
        keys.sort()
        max_len = max( len(i) for i in keys )+2


        for i in keys:
            neededlength = max_len-len(i)
            whitespace   = ' '*neededlength

            try:
                command+="   ** {0}{1}= {2:.4f}\n".format(i,whitespace,self.__dict__[i])
            except ValueError:
                continue

        return command

## THE END ##
