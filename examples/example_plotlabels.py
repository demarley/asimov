from array import array


def hist1d(nbins,bin_low,bin_high):
    """
    Set the binning for a given histogram.
    @param nbins	  Number of bins in histogram
    @param bin_low    Lower bin edge
    @param bin_high   Upper bin edge
    """
    binsize = float(bin_high-bin_low)/nbins
    arr     = array('d',[i*binsize+bin_low for i in xrange(nbins+1)])
    return arr


class Sample(object):
    """Class for organizing plotting information about physics samples"""
    def __init__(self,label='',color=''):
        self.label = label
        self.color = color

class Variable(object):
    """Class for organizing plotting information about variables"""
    def __init__(self,binning=[],label=''):
        self.binning = binning
        self.label   = label


def variable_labels():
    """Dictionaries that contain Variables objects."""
    variables = {}

    variables['mass_MMC']           = Variable(binning=hist1d(50,0,500),     label=r'MMC m$_\text{H}$ [GeV]')
    variables['mass_vis']           = Variable(binning=hist1d(20,0,200),     label=r'm$_{\tau\ell}$ [GeV]')
    variables['pt_h']               = Variable(binning=hist1d(30,0,300),     label=r'p$_\text{T}$(H) [GeV]')
    variables['deltaeta_jet_jet']   = Variable(binning=hist1d(10,0,5),       label=r'$\Delta\eta$(j$_\text{1}$,j$_\text{2}$)')
    variables['mass_jet_jet']       = Variable(binning=hist1d(100,0,2000),   label=r'm$_{\text{j}_{\text{1}}\text{,j}_{\text{2}}}$')
    variables['prodeta_jet_jet']    = Variable(binning=hist1d(72,-18,18),    label=r'$\eta_{\text{j}_\text{1}}\times\eta_{\text{j}_\text{2}}$')
    variables['deltar_tau_lep']     = Variable(binning=hist1d(10,0,5),       label=r'$\Delta$R($\tau$,$\ell$)')
    variables['pt_tot']             = Variable(binning=hist1d(50,0,500),     label=r'p$_\text{T}$(total)')
    variables['sum_pt']             = Variable(binning=hist1d(100,0,1000),   label=r'S$_\text{T}$')
    variables['pt_ratio_lep_tau']   = Variable(binning=hist1d(20,0,10),      label=r'p$_\text{T}^\ell$/p$_\text{T}^\tau$')
    variables['met_phi_centrality'] = Variable(binning=hist1d(30,-1.5,1.5),  label=r'$\phi^\text{miss}_\text{c}$')
    variables['lep_eta_centrality'] = Variable(binning=hist1d(10,0,1.5),     label=r'$\eta^\ell_\text{c}$')
    variables['mass_transverse_met_lep'] = Variable(binning=hist1d(20,0,200),label=r'm$_\text{T}$')

    return variables



def sample_labels():
    """Dictionaries that contain Samples objects.
       > The key values match those in config/sampleMetadata.txt.
         (The functions in util.py are setup to read the information this way.)
         If you want something unique, then you may need to specify 
         it manually in your plotting script
    """
    ## Sample information
    samples = {}

    samples['signal'] = Sample(label='Signal',color='b')
    samples['bckg']   = Sample(label='Bckg',color='r')

    return samples
