"""
Created:        16 August 2018
Last Updated:   16 August 2018

Dan Marley
daniel.edison.marley@cernSPAMNOT.ch
Texas A&M University
-----

Base class for plotting deep learning information & performance

Designed for running on desktop at TAMU
with specific set of software installed
--> not guaranteed to work in CMSSW environment!
"""
import os
import sys
import json
import itertools
from datetime import date
from collections import OrderedDict

# load hepPlotter code (applies matplotlib settings)
try:
    CMSSW_BASE = os.environ['CMSSW_BASE']
    from Analysis.hepPlotter.histogram1D import Histogram1D
    from Analysis.hepPlotter.histogram1D import Histogram2D
    import Analysis.hepPlotter.labels as hpl
    import Analysis.hepPlotter.tools as hpt
except KeyError:
    cwd = os.getcwd()
    hpd = cwd.replace("asimov","hepPlotter/python/")
    if hpd not in sys.path:
        sys.path.insert(0,hpd)
    from histogram1D import Histogram1D
    from histogram2D import Histogram2D
    import labels as hpl
    import tools as hpt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import util



class Empire(object):
    """Plotting utilities for deep learning"""
    def __init__(self):
        """Give default values to member variables"""
        self.formatter       = FormatStrFormatter('%g')         # formatting axis labels
        self.betterColors    = hpt.betterColors()['linecolors'] # colors for plotting
        self.sample_labels   = {}                               # Formatted sample labels
        self.variable_labels = {}                               # Formatted variable labels

        self.msg_svc      = util.VERBOSE()  # 'level' for printing statements
        self.output_dir   = ''              # directory/path to store the plots
        self.image_format = 'pdf'           # figure format (PDF matches with backend!)
        self.class_pairs  = None            # Unique pairs of targets for comparing
        self.features     = []              # Features used in network

        self.classCollection = util.NNClassCollection()  # NN classes in deep learning
        self.CMSlabelStatus  = "Simulation Internal"     # plot label 


    def initialize(self,nn_classes):
        """
        Set parameters of class to make plots

        @param nn_classes       Collection of NNClass() objects
        """
        self.featurePairs = list(itertools.combinations(self.features,2))

        self.classCollection.clear()
        for i,nc in enumerate(nn_classes):
            tmp = nc
            tmp.label = self.sample_labels[nc.name].label
            tmp.color = self.betterColors[i]
            self.classCollection.append(tmp)

        # Create unique combinations of the targets in pairs 
        # (to calculate separation between classes in two dimensions)
        self.class_pairs = list(itertools.combinations(self.classCollection.names(),2))

        return


    def feature(self,dataframe,ndims=-1):
        """
        Plot the features
          For classification, compare different targets
          For regression, just plot the features

        @param dataframe    Pandas dataframe with data to be plotted
        @param ndims        Number of dimensions to plot (-1=ALL; 1=1D features only) 
                            [always plot 1D features, for now]
        """
        self.msg_svc.DEBUG("DL : Plotting features.")

        self.separations = dict( (k,{}) for k in self.features)
        for featurepairs in self.featurePairs:
             self.separations['-'.join(featurepairs)] = {}

        ## ++ Plot the features for each target
        for hi,feature in enumerate(self.features):
            vl = self.variable_labels[feature]

            hist = Histogram1D()

            hist.normed  = True
            hist.stacked = False
            hist.binning = vl.binning
            hist.x_label = vl.label
            hist.y_label = "A.U." if hist.normed else "Events"
            hist.format  = self.image_format
            hist.saveAs  = self.output_dir+"/hist_"+feature
            hist.CMSlabel = 'outer'
            hist.CMSlabelStatus = self.CMSlabelStatus
            hist.legend['fontsize'] = 10

            hist.ratio.value  = "significance"
            hist.ratio.ylabel = r"Sig."
            hist.ratio.update_legend = True

            hist.initialize()

            # Draw the distribution for this feature for each NN class
            histValues = {}
            for c in self.classCollection:
                kwargs = {"draw_type":"step","edgecolor":c.color,"label":c.label}

                # Put into histogram before passing to hepPlotter to reduce memory
                h,bx = np.histogram(dataframe[dataframe.target==c.value][feature],bins=vl.binning)
                bin_centers = 0.5*(bx[:-1]+bx[1:])
                hist.Add(bin_centers,weights=h,name=c.name,**kwargs)
                histValues[c.name] = h.copy()

            # Add ratio plot comparing the targets (in pairs) for this feature
            # e.g., feature (QCD) vs feature (QB), etc.
            numerators = {}
            markers = ['o','v','^']         # for ratios with the same numerator, change the marker style
            for pair in self.class_pairs:
                try:
                    idx = numerators[pair[0]]
                    numerators[pair[0]]+=1
                except KeyError:
                    idx = 0
                    numerators[pair[0]]=1
                num = self.classCollection.get(pair[0])
                den = self.classCollection.get(pair[1])
                hist.ratio.Add(numerator=pair[0],denominator=pair[1],draw_type='errorbar',
                               mec=num.color,mfc=num.color,ecolor=num.color,fmt=markers[idx],
                               label=r"%s/$\sqrt{\text{%s}}$"%(num.label,den.label))

            p = hist.execute()
            hist.savefig()

            ## Calculate 1D separations for this feature between classes
            for pair in self.class_pairs:
                data_a = histValues.get(pair[0])
                data_b = histValues.get(pair[1])
                separation = util.getSeparation(data_a,data_b)
                self.separations[feature]['-'.join(pair)] = separation

        ## ++ Plot two features against each other for each target (multi-jet,W,QB,tt_bckg)
        for hi,featurepairs in enumerate(self.featurePairs):
            xfeature = featurepairs[0]
            yfeature = featurepairs[1]

            xvar = self.variable_labels[xfeature]
            yvar = self.variable_labels[yfeature]
            xbins = xvar.binning
            ybins = yvar.binning

            histValues = {}
            for c in self.classCollection:
                # save memory by making the histogram here and passing the result to hepPlotter
                xdf = dataframe[dataframe.target==c.value][xfeature]
                ydf = dataframe[dataframe.target==c.value][yfeature]
                h,binsx,binsy = np.histogram2d(xdf,ydf,bins=[xbins,ybins])
                histValues[c.name] = h
                # h[0] yields the y-axis array for the first bin in x

                if ndims==1: return  # only plot 1D features

                hist = Histogram2D()

                hist.colormap = 'default'
                hist.colorbar['title'] = "Events"

                try:
                    hist.binning = [xbins.tolist(),ybins.tolist()]
                except:
                    hist.binning = [xbins,ybins]
                hist.x_label = xvar.label
                hist.y_label = yvar.label
                hist.format  = self.image_format
                hist.saveAs  = self.output_dir+"/hist2d_"+c.name+"_"+xfeature+"-"+yfeature
                hist.CMSlabel = 'outer'
                hist.CMSlabelStatus = self.CMSlabelStatus
                hist.logplot['data'] = True

                hist.extra_text.Add(c.label,coords=[0.03,0.97])

                hist.initialize()

                # create dummy binning
                binsx  = 0.5*(binsx[:-1]+binsx[1:])
                binsy  = 0.5*(binsy[:-1]+binsy[1:])
                xdummy = binsx.repeat(len(binsy))
                ydummy = np.tile(binsy, (1,len(binsx)) )[0]

                hist.Add([xdummy,ydummy],weights=h.flatten(),name=c.name)

                p = hist.execute()
                hist.savefig(dpi=100)

            ## Calculate 2D separations for these features between classes
            for pair in self.class_pairs:
                data_a = histValues[pair[0]]
                data_b = histValues[pair[1]]
                separation = util.getSeparation(data_a,data_b)
                self.separations['-'.join(featurepairs)]['-'.join(pair)] = separation

        ## ++ Save separation info to CSV file
        #     Storing raw values of separations to plot / analyze later
        for pair in self.class_pairs:
            saveAs1 = "{0}/separations1D_{1}-{2}".format(self.output_dir,pair[0],pair[1])
            saveAs2 = "{0}/separations2D_{1}-{2}".format(self.output_dir,pair[0],pair[1])

            fcsv1 = open("{0}.csv".format(saveAs1),"w")
            fcsv2 = open("{0}.csv".format(saveAs2),"w")

            fcsv1.write("feature,separation")
            fcsv2.write("xfeature,yfeature,separation")

            for f in self.separations.keys():
                separation = self.separations[f]['-'.join(pair)]
                if '-' in f:
                    feature_x,feature_y = f.split('-')
                    fcsv2.write("{0},{1},{2}".format(feature_x,feature_y,separation))
                else:
                    fcsv1.write("{0},{1}".format(f,separation))

            fcsv1.close() 
            fcsv2.close() 


        return


    def separation(self):
        """Plot the separations between classes of the NN for different features"""
        self.msg_svc.DEBUG("DL : Plotting separations.")

        listOfFeatures     = list(self.features)
        listOfFeaturePairs = list(self.featurePairs)
        featurelabels      = [self.variable_labels[f].label for f in self.features]

        nfeatures = len(listOfFeatures)

        for target in self.class_pairs:
            target_a = target[0]
            target_b = target[1]

            ##                                 ##
            ## One dimensional separation plot ##
            ##  - horizontal bar chart         ##
            ##                                 ##
            saveAs = "{0}/separations1D_{1}-{2}".format(self.output_dir,target_a,target_b)

            separations = [self.separations[f]['-'.join(target)] for f in listOfFeatures]

            # sort data by separation value
            data = list( zip(listOfFeatures,separations) )
            data.sort(key=lambda x: x[1])
            listOfFeatures[:],separations[:] = zip(*data)

            # make the bar plot
            fig,ax = plt.subplots()
            ax.barh(listOfFeatures, separations, align='center')
            ax.set_yticks(listOfFeatures)
            ax.set_yticklabels(featurelabels,fontsize=12)
            ax.set_xticklabels([self.formatter(i) for i in ax.get_xticks()])
            ax.set_xlabel("Separation")

            # CMS/COM Energy Label + Signal name
            self.stamp_cms(ax)
            self.stamp_energy(ax)
            ax.text(0.95,0.05,"{0} - {1}".format(target_a,target_b),fontsize=16,
                    ha='right',va='bottom',transform=ax.transAxes)

            plt.savefig("{0}.{1}".format(saveAs,self.image_format))
            plt.close()


            ##                                 ##
            ## Two dimensional separation plot ##
            ##                                 ##
            ##  from the separation values for each unique (xfeature,yfeature) combination
            ##  build a matrix that can be drawn using hist2d()
            saveAs  = "{0}/separations2D_{1}-{2}".format(self.output_dir,target_a,target_b)
            x_coord = [self.features.index(f[0]) for f in listOfFeaturePairs]
            y_coord = [self.features.index(f[1]) for f in listOfFeaturePairs]
            separations = [self.separations['-'.join(f)]['-'.join(target)] for f in listOfFeaturePairs]

            # Now repeat the entries with flipped indices to get the full matrix
            x = list(x_coord)+list(y_coord)
            y = list(y_coord)+list(x_coord)
            separations += separations

            # make the plot
            hist = Histogram2D()

            hist.colormap = 'default'
            hist.colorbar['title'] = "Separation"

            hist.x_label = None
            hist.y_label = None
            hist.binning = [range(nfeatures+1),range(nfeatures+1)]
            hist.format  = self.image_format
            hist.saveAs  = saveAs
            hist.CMSlabel = 'outer'
            hist.CMSlabelStatus = self.CMSlabelStatus

            hist.initialize()
            hist.Add([x,y],weights=separations,name='-'.join(target))

            fig = hist.execute()

            # shift location of ticks to center of the bins
            ax = fig.gca()
            ax.set_xticks(np.arange(nfeatures)+0.5, minor=False)
            ax.set_yticks(np.arange(nfeatures)+0.5, minor=False)
            ax.set_xticklabels(featurelabels, minor=False, ha='right', rotation=70, fontsize=12)
            ax.set_yticklabels(featurelabels, minor=False, fontsize=12)

            hist.savefig()

        return


    def correlation(self,corrmats={}):
        """Plot correlations between features of the NN"""
        self.msg_svc.DEBUG("DL : Plotting correlations.")

        opts = {'cmap':plt.get_cmap("bwr"),'vmin':-1,'vmax':1}

        for c in self.classCollection:
            saveAs = "{0}/correlations_{1}".format(self.output_dir,c.name)

            corrmat = corrmats[c.name]

            # Save correlation matrix to CSV file
            corrmat.to_csv("{0}.csv".format(saveAs))

            # Plot correlation matrix
            # -- Use matplotlib directly
            fig,ax = plt.subplots()

            heatmap1 = ax.pcolor(corrmat, **opts)
            cbar     = plt.colorbar(heatmap1, ax=ax)
            cbar.ax.set_yticklabels([i.get_text().replace("$","") for i in cbar.ax.get_yticklabels()])

            labels = [self.variable_labels[feat].label for feat in corrmat.columns.values]
            # shift location of ticks to center of the bins
            ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
            ax.set_xticklabels(labels, fontsize=12, minor=False, ha='right', rotation=70)
            ax.set_yticklabels(labels, fontsize=12, minor=False)

            ## CMS/COM Energy Label + Signal name
            self.stamp_cms(ax)
            self.stamp_energy(ax)
            ax.text(0.03,0.93,c.label,fontsize=16,ha='left',va='bottom',transform=ax.transAxes)

            plt.savefig("{0}.{1}".format(saveAs,self.image_format),
                        format=self.image_format,dpi=300,bbox_inches='tight')
            plt.close()

        return



    def prediction(self,train_data={},test_data={}):
        """
        Plot the training and testing predictions. 
        To save on memory, pass this histograms directly
        
        Multi-classification:
          Make a plot for each target value 
          (e.g., QCD prediction to be QCD; Top prediction to be QCD, etc)
          Need two-dimensional arrays/dictionaries to achieve this
        """
        self.msg_svc.DEBUG("DL : Plotting DNN prediction. ")

        for c in self.classCollection:
            target_label = self.sample_labels[c.name].label
            hist = Histogram1D()

            hist.normed  = True  # compare shape differences (likely don't have the same event yield)
            hist.format  = self.image_format
            hist.saveAs  = "{0}/hist_DNN_prediction_{1}".format(self.output_dir,c.name)
            hist.stacked = False
            hist.x_label = "Prediction: {0}".format(target_label)
            hist.y_label = "A.U."
            hist.CMSlabel = 'outer'
            hist.CMSlabelStatus   = self.CMSlabelStatus
            hist.legend['fontsize'] = 10

            hist.ratio.value  = "ratio"
            hist.ratio.ylabel = "Train/Test"

            hist.initialize()

            json_data = {}
            for t,cc in enumerate(self.classCollection):

                target_value = cc.value  # arrays for multiclassification 

                train_t = train_data[c.name][cc.name]
                train_weights = train_t[0]
                train_bins    = train_t[1]
                train_dummy   = hpt.midpoints(train_bins)
                train_kwargs  = {"draw_type":"step","edgecolor":cc.color,
                                 "label":cc.label+" Train"}

                test_t  = test_data[c.name][cc.name]
                test_weights = test_t[0]
                test_bins    = test_t[1]
                test_dummy   = hpt.midpoints(test_bins)
                test_kwargs  = {"draw_type":"stepfilled","edgecolor":cc.color,
                                "color":cc.color,"linewidth":0,"alpha":0.5,
                                "label":cc.label+" Test"}

                hist.binning = train_bins  # should be the same for train/test

                hist.Add(train_dummy,weights=train_weights,\
                         name=cc.name+'_train',**train_kwargs) # Training
                hist.Add(test_dummy,weights=test_weights,\
                         name=cc.name+'_test',**test_kwargs)    # Testing

                hist.ratio.Add(numerator=cc.name+'_train',denominator=cc.name+'_test')

                ## Save data to JSON file
                json_data[cc.name+"_train"] = {"binning":train_t[1].tolist(),
                                               "content":train_t[0].tolist()}
                json_data[cc.name+"_test"]  = {"binning":test_t[1].tolist(),
                                               "content":test_t[0].tolist()}

            p = hist.execute()
            hist.savefig()

            # calculate separation between predictions
            separations = OrderedDict()
            for t,target in enumerate(self.class_pairs):
                data_a = json_data[ target[0]+"_test" ]["content"]
                data_b = json_data[ target[1]+"_test" ]["content"]
                separation = util.getSeparation(np.asarray(data_a),np.asarray(data_b))
                json_data[ '-'.join(target)+"_test" ] = {"separation":separation}
                separations['-'.join(target)] = separation

            # save results to JSON file (just histogram values & bins) to re-make plots
            with open("{0}.json".format(hist.saveAs), 'w') as outfile:
                json.dump(json_data, outfile)


            ## Plot separation between predictions for given target
            saveAs = "{0}/hist_DNN_prediction_sep_{1}".format(self.output_dir,c.name)
            sorted_sep = sorted(separations, key=separations.__getitem__) # sort data by separation value
            ypos = np.arange(len(sorted_sep))

            # make the bar plot
            fig,ax = plt.subplots()
            ax.barh(ypos, [separations[i] for i in sorted_sep], align='center')
            ax.set_yticks(ypos)
            yticklabels = []
            for i in sorted_sep:
                split  = i.split("-")
                first  = self.sample_labels[ split[0] ].label
                second = self.sample_labels[ split[1] ].label
                yticklabels.append( '{0}-{1}'.format(first,second) )
            ax.set_yticklabels(yticklabels,fontsize=12)
            ax.set_xticklabels([self.formatter(i) for i in ax.get_xticks()])
            ax.set_xlabel("Separation",ha='right',va='top',position=(1,0))

            # CMS/COM Energy Label + Signal name
            self.stamp_cms(ax)
            self.stamp_energy(ax)
            ax.text(0.95,0.05,"DNN Prediction: {0}".format(target_label),fontsize=16,
                    ha='right',va='bottom',transform=ax.transAxes)

            plt.savefig("{0}.{1}".format(saveAs,self.image_format))
            plt.close()

        return



    def ROC(self,fprs={},tprs={},roc_auc={}):
        """Plot the ROC curve & save to text file"""
        self.msg_svc.DEBUG("DL : Plotting ROC curve.")

        saveAs = "{0}/roc_curve".format(self.output_dir)

        ## Use matplotlib directly
        fig,ax = plt.subplots()

        # Draw all of the ROC curves from the K-fold cross-validation
        ax.plot([0,1],[0,1],ls='--',label='No Discrimination',lw=2,c='gray')
        ax.axhline(y=1,lw=1,c='k',ls='-')

        # Plot ROC curve
        for key in fprs.keys():
            label = self.sample_labels[key].label
            ax.plot(fprs[key],tprs[key],label='{0} (AUC={1:.2f})'.format(label,roc_auc[key]),lw=2)
            # save ROC curve to CSV file (to plot later)
            csv = [ "{0},{1}".format(fp,tp) for fp,tp in zip(fprs,tprs) ]
            util.to_csv("{0}.csv".format(saveAs),csv)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.5])

        ax.set_xlabel(r'Background $\epsilon$',ha='right',va='top',position=(1,0))
        ax.set_ylabel(r'Signal $\epsilon$',ha='right',va='bottom',position=(0,1))

        ax.set_xticklabels([self.formatter(i) for i in ax.get_xticks()],fontsize=20)
        ax.set_yticklabels([self.formatter(i) for i in ax.get_yticks()],fontsize=20)

        ## CMS/COM Energy Label
        self.stamp_cms(ax)
        self.stamp_energy(ax)

        leg = ax.legend(fontsize=12)
        leg.draw_frame(False)

        plt.savefig('{0}.{1}'.format(saveAs,self.image_format))
        plt.close()

        return


    def plot_history(self,history,ax=None,key='loss',index=-1):
        """Draw history of model"""
        try:
            loss     = history.history[key]
            val_loss = history.history.get('val_'+key)
        except:
            loss     = history
            val_loss = None

        x = range(1,len(loss)+1)
        label = key.title()
        if index>=0: label += ' {0}'.format(index)
        ax.plot(x,loss,label=label)
        csv = [ "{0},{1}\n".format(i,j) for i,j in zip(x,loss) ]

        if val_loss is not None:
            label = 'Validation {0}'.format(index) if index>=0 else 'Validation'
            ax.plot(x,val_loss,label=label)
            csv += [ "{0},{1}\n".format(i,j) for i,j in zip(x,val_loss) ]

        return csv


    def history(self,history,kfold=-1):
        """Plot history as a function of epoch for model"""
        self.msg_svc.DEBUG("DL : Plotting loss as a function of epoch number.")

        for key in ['loss','acc']:
            fig,ax = plt.subplots()

            saveAs   = "{0}/history_{1}".format(self.output_dir,key)
            csv      = self.plot_history(history,ax=ax,key=key)
            filename = "{0}.csv".format(saveAs)
            util.to_csv(filename,csv)

            ax.set_xlabel('Epoch',fontsize=22,ha='right',va='top',position=(1,0))
            ax.set_ylabel(key.title(),fontsize=22,ha='right',va='bottom',position=(0,1))

            ax.set_xticklabels([self.formatter(i) for i in ax.get_xticks()],fontsize=20)
            ax.set_yticklabels(['']+[self.formatter(i) for i in ax.get_yticks()[1:-1]]+[''],fontsize=20)

            ## CMS/COM Energy Label
            self.stamp_cms(ax)
            self.stamp_energy(ax)

            leg = ax.legend(loc=0,numpoints=1,fontsize=12,ncol=1,columnspacing=0.3)
            leg.draw_frame(False)

            plt.savefig(self.output_dir+'/{0}_epochs.{1}'.format(key,self.image_format),
                        format=self.image_format,bbox_inches='tight',dpi=200)
            plt.close()

        return



    def stamp_energy(self,axis,ha='right',coords=[0.99,1.00],fontsize=16,va='bottom'):
        energy_stamp = hpl.EnergyStamp()
        axis.text(coords[0],coords[1],energy_stamp.text,fontsize=fontsize,ha=ha,va=va,transform=axis.transAxes)
        return

    def stamp_cms(self,axis,ha='left',va='bottom',coords=[0.02,1.00],fontsize=16):
        cms_stamp = hpl.CMSStamp(self.CMSlabelStatus)
        axis.text(coords[0],coords[1],cms_stamp.text,fontsize=fontsize,ha=ha,va=va,transform=axis.transAxes)
        return


## THE END ##
