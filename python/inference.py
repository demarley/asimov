"""
Created:         6 September 2018
Last Updated:    6 September 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Simple implementation of inference-only setup in Asimov.
Likely needs to be extended by users.
"""
import json
import datetime

import util
from foundation import Foundation

import uproot
import numpy as np
import pandas as pd


# fix random seed for reproducibility
seed = 2018
np.random.seed(seed)


class Inference(Foundation):
    """Deep Learning base class"""
    def __init__(self):
        super(Inference,self).__init__()


    def initialize(self):
        """  """
        super(Inference,self).initialize()

        self.load_model()

        return


    def inference(self,data=None):
        """Run inference of the NN model"""
        if data is None:
            try:
                data = self.df[self.features].values
            except:
                self.msg_svc.ERROR("INFERENCE : inference() cannot proceed:")
                self.msg_svc.ERROR("INFERENCE : - 'data' is None and no HEP data")
                self.msg_svc.ERROR("INFERENCE : Please check your implementation.")
                return -999

        return self.predict(data)


## THE END ##