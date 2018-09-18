"""
Created:        --
Last Updated:    6 September 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

File that holds any and all misc. functions / objects
to be called from other python scripts.
(All information in one file => one location to update!)
"""
import numpy as np


class NNClass(object):
    """Class to contain information for classes used in training/inference"""
    def __init__(self,name=""):
        self.name  = name     # Name of this target, e.g., 'signal'
        self.df    = None     # dataframe of this target's features
        self.color = 'k'
        self.label = ''
        self.value = -999
        self.binning = 1


class NNClassCollection(object):
    """
    Container for multiple NN classes.
    Similar to a list with ability to access data like a dictionary 
    https://stackoverflow.com/questions/21665485/how-to-make-a-custom-object-iterable
    """
    def __init__(self):
        self.clear()

    def names(self):
        return self.m_names

    def clear(self):
        """Clear contents"""
        self.m_names = list()
        self.m_collections = list()

    def append(self,item):
        """Add a new class to the collection"""
        self.m_collections.append(item)
        self.m_names.append(item.name)

    def get(self,key):
        """Access member of the collection using the name"""
        index = self.m_names.index(key)
        return self.m_collections[index]

    # iterator attributes
    def __iter__(self):
        self.__i = 0
        return iter(self.m_collections)

    def __next__(self):
        if self.__i<len(self.m_collections)-1:
            self.__i += 1
            return self.m_collections[self.__i]
        else:
            raise StopIteration




def getHistSeparation( sig,bkg ):
    """Compare TH1* S and B -- need same dimensions
       Copied from : https://root.cern.ch/doc/master/MethodBase_8cxx_source.html#l02740
    """
    assert sig.GetNbins()==bkg.GetNbins()

    separation = 0

    nbinsx = sig.GetNbinsX()
    nbinsy = sig.GetNbinsY()

    integral_s = sig.Integral()
    integral_b = bkg.Integral()

    for x in range(nbinsx):
        for y in range(nbinsy):
            s = sig.GetBinContent( x+1,y+1 )/integral_s
            b = bkg.GetBinContent( x+1,y+1 )/integral_b

            if (s+b)!=0: separation += (s-b)**2/(s+b)

    separation *= 0.5

    return separation



def getSeparation( sig,bkg ):
    """Calculate separation between two numpy arrays (any dimension!)"""
    try:
        matching_shapes = sig.shape==bkg.shape
    except AttributeError:
        matching_shapes = len(sig)==len(bkg)
    sig_sum  = np.all(sig==0)
    bkg_sum  = np.all(bkg==0)
    zero_sum = np.all(sig+bkg==0)

    if not matching_shapes: return -1
    if zero_sum or sig_sum or bkg_sum: return -1

    sig = np.divide(sig,np.sum(sig),dtype=np.float32)
    bkg = np.divide(bkg,np.sum(bkg),dtype=np.float32)
    tmp = np.divide( (sig-bkg)**2 , (sig+bkg), dtype=np.float32)
    tmp = np.nan_to_num(tmp)    # set NaN to 0; INF to large number
    separation = tmp.sum()*0.5

    return separation



def read_config(filename,delimiter=" ",comment="#"):
    """
    Read configuration file with data stored like:
       'config option'
    And the 'config' and 'option' are separated by a character, e.g., " "
    Any lines that start with the comment are ignored
    Any text after a comment is ignored (extra white space removed)
    """
    data = file2list(filename)

    cfg = {}
    for line in data:
        line  = line.split(comment)[0]              # ignore comments on a given line
        if not line: continue                       # skip lines with comments
        line  = line.rstrip(' ')                    # remove extra whitespace
        ldata = line.split(delimiter)

        cfg[ldata[0]] = ldata[1]

    return cfg


def extract(str_value, start_='{', stop_='}'):
    """Extract a string between two symbols, e.g., parentheses."""
    extraction = str_value[str_value.index(start_)+1:str_value.index(stop_)]
    return extraction


def to_csv(filename,data):
    """Write data to CSV file"""
    if not filename.endswith(".csv"): filename += ".csv"
    f = open(filename,"w")
    for d in data:
        f.write(d)
    f.close()

    return


def file2list(filename):
    """Load text file and dump contents into a list"""
    listOfFiles = open( filename,'r').readlines()
    listOfFiles = [i.rstrip('\n') for i in listOfFiles if not i.startswith("#")]
    return listOfFiles


def str2bool(param):
    """Convert a string to a boolean"""
    return (param in ['true','True','1'])




class VERBOSE(object):
    """Object for handling output"""
    def __init__(self):
        self.verboseMap = {"DEBUG":  0,
                           "INFO":   1,
                           "WARNING":2,
                           "ERROR":  3,
                           "MUTE":   4};
        self.level     = "INFO"
        self.level_int = self.verboseMap[self.level]

    def initialize(self):
        """Setup the integer level value"""
        self.level_int = self.verboseMap[self.level]

    def level_value(self):
        """Return the integer value"""
        return self.level_int

    def DEBUG(self,message):
        """Debug level - most verbose"""
        self.verbose("DEBUG",message)
        return

    def INFO(self,message):
        """Info level - standard output"""
        self.verbose("INFO",message)
        return

    def WARNING(self,message):
        """Warning level - if something seems wrong but code can continue"""
        self.verbose("WARNING",message)
        return

    def ERROR(self,message):
        """Error level - something is wrong"""
        self.verbose("ERROR",message)
        return

    def compare(self,level1):
        """Compare two levels"""
        return self.verboseMap[level1]>=self.level_int
            

    def verbose(self,level,message):
        """Print message to the screen"""
        if self.compare( level ):
            print " {0} :: {1}".format(level,message)
        return


## THE END ##
