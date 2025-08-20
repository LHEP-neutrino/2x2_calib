import sys
import subprocess
import numpy as np
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
import logging

# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def set_log_level(level_name: str):
    """
    Change log level at runtime.

        input: level_name, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    """
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)
        logger.info(f"Log level changed to {level_name.upper()}")
    else:
        logger.error(f"Invalid log level: {level_name}")


class calibWvfms:
    ''' 
        Class to test and set up a calibration routine for the FSD waveforms

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            - dumpWvfms()           :   Dump the waveforms as png at the output path
            
    '''

    # Initialize the class
    def __init__(self, filedir, filename, output_path=None, log_level=None):

        if log_level != None:
            set_log_level(log_level)
        
        # Open files
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')
        else:
            self.output_path = os.path.abspath(output_path)


        # self.run_info = f['run_info']
        # self.is_mc = self.run_info.attrs['is_mc']

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']
        # self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
        # self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

        # self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
        # self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
        # self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

        # self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
        # self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
        # self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
        # self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        self.Npeaks = np.zeros(self.light_wvfms.shape[:-1])

        
        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        # Information about the selection:
        print(f'Processing file {filedir+filename}')
        print(f'The output path is set to {self.output_path}')
        print(f"Number of events in the selection: {len(self.light_events)}")

    def _extract_raiseEdge(self, wvfm, threshold):
        '''
        threshold in ADC unit
        '''
        start_idx = []
        # end

        Npt_beforeThresold = 2

        for ix in range(Npt_beforeThresold, len(wvfm)):
            if (wvfm[ix-1] < threshold and wvfm[ix-2] < threshold and wvfm[ix] >= threshold):
                start_idx.append(ix)

            # if (wvfm[ix-1] > threshold and wvfm[ix-2] > threshold and wvfm[ix] <= threshold):
        

        return start_idx
    
    def _extract_peak(self, wvfm, minWidth, verbose=False):
        '''
        minWidth: Minimal width of a peak (e.g. a 5 ticks peaks have two values lower than the peak summit on each side)
        '''
        peak_idx = []
        is_peak = True
        mean = np.mean(wvfm)

        Npt_peakSide = int(minWidth/2)


        for ix in range(Npt_peakSide+1, len(wvfm)-Npt_peakSide-1):
            for i in range(1, Npt_peakSide+1):
                if (wvfm[ix-i] > wvfm[ix] or wvfm[ix+i] > wvfm[ix]):
                    is_peak = False
                    break

            if (is_peak==True):
                peak_idx.append(ix)
            else:
                is_peak = True

        peak_idx = np.array(peak_idx)
        
        if (verbose==False):
           aboveMean_idx = np.where(wvfm[peak_idx]>mean)[0]
           peak_idx = peak_idx[aboveMean_idx]

        return peak_idx, mean 
    

    
    def findPeak_wvfms(self, event, adc, chan, minWidth=5, verbose=False):
        '''
        Find the number of peak and store it in self.Npeaks

        Args:
            
            
        Return:
            None
        '''

        peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose)
        
        if (verbose==True):
            print(f'{len(peak_idx)} peaks were found with a minWidth of {minWidth}.')

        # if (verbose==True):
        #     mean = peak_info[0]
        #     aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > mean)[0]]
        #     belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= mean)[0]]
        #     ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='')
        #     ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='')

        #     ax_wvfm.hlines(peak_info[0], xlim[0], xlim[1], color='r', ls='--')

        #     self.Npeaks[event][adc,chan]=len(aboveMean_idx)

        
        self.Npeaks[event][adc,chan]=len(peak_idx)

        return None


    def plot_wvfm(self, event, adc, chan, xlim=None, threshold=None, peakFinder=False, minWidth=5, baseline=None, verbose=False, show_plot=False, output=None):
        '''
        Plot the light waveforms.

        Args:
            
            
        Return:
            None
        '''
        # Compute the x-axis coordinate
        x_ticks = np.arange(0, self.light_wvfms.shape[-1],  1)

        # print(x_ticks)

        # Setup the plot
        fig_wvfm = plt.figure(figsize=[12.8, 4.8])
        ax_wvfm = fig_wvfm.subplots()
        
        if (xlim is None):
            xlim = [x_ticks[0], x_ticks[-1]+1]

        ax_wvfm.set_xticks(np.arange(xlim[0], xlim[-1]+50, 50))
        ax_wvfm.set_xlim(xlim)
        ax_wvfm.set_xlabel('ticks')
        ax_wvfm.set_ylabel('ADC value')
        ax_wvfm.grid(True)

        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='.', ls='', label=f' Data points')# Event {event}, ADC {adc}, Chan. {chan}')
        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='', ls='-', c='r', alpha=0.3)

        if (threshold != None):
            start_idx = self._extract_raiseEdge(self.light_wvfms[event][adc,chan], threshold) 
            ylim = ax_wvfm.get_ylim()
            print(f' There are {len(start_idx)} triggers with a threshold at {threshold}')
            for idx in start_idx:
                ax_wvfm.vlines(idx, ylim[0], ylim[1], color='k', ls='--')

            ax_wvfm.hlines(threshold, xlim[0], xlim[1], color='r', ls='--')

        if (peakFinder==True):
            peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose)
            mean = peak_info[0]

            if (verbose==True):
                aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > mean)[0]]
                belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= mean)[0]]
                ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='', label='Accepted peaks')
                ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='Discarded peaks')

                ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')

                self.Npeaks[event][adc,chan]=len(aboveMean_idx)

                print(f'{len(aboveMean_idx)} peaks were found above the mean with a minWidth of {minWidth} and {len(belowMean_idx)} were cut out.')

            else:
                ax_wvfm.plot(peak_idx, self.light_wvfms[event][adc,chan][peak_idx], color='g', marker='x', ls='', label='Peaks found')
                ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')
                self.Npeaks[event][adc,chan]=len(peak_idx)


        if (baseline != None):
            ax_wvfm.hlines(baseline, xlim[0], xlim[1], color='k', ls='--', label='Baseline')

        ax_wvfm.legend()
        ax_wvfm.set_title(f'Waveforms of Event {event} for ADC {adc}, Chan. {chan}')

        if isinstance(output, PdfPages):
            output.savefig()
            plt.close()
        elif output is not None:
            fig_wvfm.savefig(output)

        if show_plot == False:
            plt.close()

        return None
    
    def plot_wvfms(self, events, adcs=None, chans=None, threshold=None, peakFinder=False, minWidth=5, baseline=None, save_plots=False, verbose=False, show_plots=False):
        inputFile_name = self.filename.split(".")[0]

        _, Nadc, Nchan, _ = self.light_wvfms.shape

        if isinstance(events, int):
            events = np.array([events, events+1])
        elif (isinstance(events, (list, tuple)) and len(events) == 2):
            events = np.array(events)
        else:
            print("Invalid 'events' input, should be int or ArrayLike")

        if adcs is None:
            adcs = np.array([0, Nadc])

        if chans is None:
            chans = np.array([0, Nchan])
            

        for i_event in range(*events):
            for j_adc in range(*adcs):
                    for k_chan in range(*chans):
                        if save_plots is None:
                            output_plot = None
                        else:
                            output_path = os.path.join(self.output_path, inputFile_name, f'adc{j_adc}', f'chan{k_chan}')
                            os.makedirs(output_path, exist_ok=True)
                            output_plot=f"{output_path}/Ev{i_event}_adc{j_adc}_chan{k_chan}_wvfm.png"

                        self.plot_wvfm(i_event, j_adc, k_chan, threshold=threshold, peakFinder=peakFinder, minWidth=minWidth, baseline=baseline, verbose=verbose, output=output_plot, show_plot=show_plots)
    
        return None
    

    
    def _plot_summaryDC(self, adc, show_plot=False, output=None, peakFinder_minWidth=5, inactive_channels=None):
        Nevents, Nadc, Nchan, Nticks = self.light_wvfms.shape
        # Compute the x-axis coordinate
        x_chan = np.arange(0, Nchan,  1)
        xlim = [x_chan[0]-0.5, x_chan[-1]+0.5]

        if inactive_channels is not None:
            mask_inactive_chans = np.isin(x_chan, inactive_channels)
            x_chan = x_chan[~mask_inactive_chans]

        Nevents = 100

        # Setup the plot
        fig_summaryDC = plt.figure(figsize=[12.8, 4.8])
        ax_summaryDC = fig_summaryDC.subplots()
        
        ax_summaryDC.set_xlim(xlim)
        ax_summaryDC.set_xticks(np.arange(xlim[0]+0.5, xlim[-1]-0.5, 2))
        
        ax_summaryDC.set_xlabel('Channel')
        ax_summaryDC.set_ylabel('mean DC rate [kHz]')
        ax_summaryDC.grid(True)

        # Compute DC rate
        # print('begin find peak')
        for i_event in range(Nevents):
            for j_chan in x_chan:
                # print(f'event {i_event}, chan {j_chan}')
                self.findPeak_wvfms(i_event, adc, j_chan, minWidth=peakFinder_minWidth)

        peaks_sum = np.sum(self.Npeaks[:,adc,x_chan], axis=0)
        # print(f' peaks sum shape{peaks_sum.shape}')
        DC_rates = peaks_sum/(Nevents*Nticks*self.time_tick)

        ax_summaryDC.plot(x_chan, DC_rates*10**-3, marker='.', ls='', label='Connected channels')
        if inactive_channels is not None:
            groups_inactive_channels = _group_inactive_channels(inactive_channels)
            ax_summaryDC.add_patch(Rectangle((groups_inactive_channels[0][0]-0.5,ax_summaryDC.get_ylim()[0]), abs(groups_inactive_channels[0][-1]-groups_inactive_channels[0][0]+1), ax_summaryDC.get_ylim()[1]-ax_summaryDC.get_ylim()[0], color='r', alpha=0.1, zorder= 0, label='Disconnected channels'))
            for group in groups_inactive_channels[1:]:
                ax_summaryDC.add_patch(Rectangle((group[0]-0.5,ax_summaryDC.get_ylim()[0]), abs(group[-1]-group[0]+1), ax_summaryDC.get_ylim()[1]-ax_summaryDC.get_ylim()[0], color='r', alpha=0.1, zorder= 0))

        ax_summaryDC.legend()
        ax_summaryDC.set_title(f'mean DC rate of ADC {adc} (over {Nevents} events)')

        if isinstance(output, PdfPages):
            output.savefig()
            plt.close()
        elif output is not None:
            fig_summaryDC.savefig(output)


        if show_plot == False:
            plt.close()


        return None

    
    def plot_summaryDC(self, events=None, adcs=None, chans=None, threshold=None, peakFinder_minWidth=5, output=None, show_plots=False, inactive_channels=None):
        inputFile_name = self.filename.split(".")[0]

        Nevents, Nadc, Nchan, _ = self.light_wvfms.shape

        if events is None:
            events = np.array([0, Nevents])
        else:
            if isinstance(events, int):
                events = np.array([events, events+1])
            elif (isinstance(events, (list, tuple)) and len(events) == 2):
                events = np.array(events)
            else:
                print("Invalid 'events' input, should be None int or ArrayLike")

        if adcs is None:
            adcs = np.array([0, Nadc])
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs, adcs+1])
            elif (isinstance(adcs, (list, tuple)) and len(adcs) == 2):
                adcs = np.array(adcs)
            else:
                print("Invalid 'adcs' input, should be None, int or ArrayLike")


        if chans is None:
            chans = np.array([0, Nchan])
            
        for j_adc in range(*adcs):
            if output is None:
                output_plot = None
            else:
                output_path = os.path.join(self.output_path, inputFile_name, f'adc{j_adc}')
                os.makedirs(output_path, exist_ok=True)
                output_plot=f"{output_path}/summaryDC_adc{j_adc}.png"
            
            self._plot_summaryDC(j_adc, show_plot=show_plots, output=output_plot, peakFinder_minWidth=peakFinder_minWidth, inactive_channels=inactive_channels)

        return None
    
    def plot_DC_pdf(self, adcs=None, chans=None, threshold=None, peakFinder_minWidth=5, output=None, show_plots=False, inactive_channels=None):
        inputFile_name = self.filename.split(".")[0]

        _, Nadc, Nchan, _ = self.light_wvfms.shape

        if adcs is None:
            adcs = np.array([0, Nadc])
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs, adcs+1])
            elif (isinstance(adcs, (list, tuple)) and len(adcs) == 2):
                adcs = np.array(adcs)
            else:
                print("Invalid 'adcs' input, should be None, int or ArrayLike")

        if chans is None:
            chans = np.array([0, Nchan])

        output_pdf = os.path.join(self.output_path, inputFile_name)
        os.makedirs(output_pdf, exist_ok=True)

        i_event = 7

        output_path = os.path.join(self.output_path, inputFile_name)
        logger.info(f'The summary pdf will be created in the folder {output_path}')

        for j_adc in range(*adcs):
            output_path_adc = os.path.join(output_path, f'adc{j_adc}')
            os.makedirs(output_path_adc, exist_ok=True)
            output_pdf=f"{output_path_adc}/summaryDC_adc{j_adc}.pdf"

            with PdfPages(output_pdf) as summary_pdf:
                self._plot_summaryDC(j_adc, show_plot=show_plots, output=summary_pdf, peakFinder_minWidth=peakFinder_minWidth, inactive_channels=inactive_channels)
                for k_chan in range(*chans):
                    self.plot_wvfm(i_event, j_adc, k_chan, threshold=threshold, peakFinder=True, minWidth=peakFinder_minWidth, output=summary_pdf, show_plot=show_plots)
    
            logger.info(f'The summary pdf of adc {j_adc} was created')
        


    def fingerPlot_Amp_wvfms(self, minWidth=5, Nbins=100):
        fig_fP_Amp = plt.figure()#figsize=[12.8, 4.8])
        ax_fP_Amp = fig_fP_Amp.subplots()
        ax_fP_Amp.set_xlabel('Peak height [ADC unit]')
        ax_fP_Amp.set_ylabel('Number of Peak')
        ax_fP_Amp.grid(True)


        amp = []

        for adc in range(1):#Nadc):
            # for chan in range(1):#Nchan):
            chan=0
            for event in range(180):
                peak_idx, *peak_info = self._extract_peak(self.light_wvfms[event][adc,chan], minWidth)
                for idx in peak_idx:
                    amp.append(self.light_wvfms[event][adc,chan][idx])

        hAmp, hAmp_bins = np.histogram(amp, bins=Nbins) 
        # Compute bin centers
        hAmp_bins_centers = (hAmp_bins[:-1] + hAmp_bins[1:]) / 2

        # Fit the function to the bin centers and counts
        p0=[700, -26900, 100, 300, -26800, 50, 150, -26600, 200]
        hAmp_popt, hAmp_pcov = curve_fit(_multigauss, hAmp_bins_centers, hAmp, p0=p0)

        ax_fP_Amp.stairs(hAmp, hAmp_bins, fill=True, zorder=5, label=f'ADC {0}, Chan. {10}')

        x_fit = np.linspace(hAmp_bins_centers.min(), hAmp_bins_centers.max(), 1000)
        ax_fP_Amp.plot(x_fit, _multigauss(x_fit, *hAmp_popt), 'r-', label='Fitted Function', zorder=10)
        print(hAmp_popt)

        ax_fP_Amp.legend()
        
        return None
    

class baselineWvfms:
    ''' 
        Class to compute the baseline for the FSD waveforms

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: save in LAr_evd/FSD_eventDisplay/ (default: None)

        Class methods:

            - dumpWvfms()           :   Dump the waveforms as png at the output path
            
    '''

    # Initialize the class
    def __init__(self, filedir, filename, output_path=None):
        
        # Open files
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'evD_{self.filename}/')
        else:
            self.output_path = os.path.abspath(output_path)


        # self.run_info = f['run_info']
        # self.is_mc = self.run_info.attrs['is_mc']

        # Load light events, waveform datasets and light geometry info if using
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']
        # self.light_event_wvfm_ref = f['light/events/ref']['light/wvfm']['ref']
        # self.light_event_wvfm_region = f['light/events/ref']['light/wvfm']['ref_region']

        # self.sipm_abs_pos = LUT.from_array(f["geometry_info/sipm_abs_pos"].attrs["meta"],f["geometry_info/sipm_abs_pos/data"])
        # self.sipm_rel_pos = LUT.from_array(f["geometry_info/sipm_rel_pos"].attrs["meta"],f["geometry_info/sipm_rel_pos/data"])
        # self.light_det_id = LUT.from_array(f["geometry_info/det_id"].attrs["meta"],f["geometry_info/det_id/data"])

        # self.all_sipm_pos = f["geometry_info/sipm_abs_pos/data"]["data"][1:]
        # self.sipm_unique_x = np.unique([pos[0] for pos in self.all_sipm_pos])
        # self.sipm_unique_z = np.unique([pos[2] for pos in self.all_sipm_pos])
        # self.sipm_unique_y = np.unique([pos[1] for pos in self.all_sipm_pos])

        self.baselines = np.zeros(self.light_wvfms.shape[:-1])

        
        # Defined some module properties
        self.N_sipm_side = int(60)
        self.N_side_tpc = 2
        self.N_tpc = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        # Information about the selection:
        print(f'Processing file {filedir+filename}')
        print(f'The output path is set to {self.output_path}')
        print(f"Number of events in the selection: {len(self.light_events)}")

    def compute_baselines(self):
        self.baselines = np.mean(self.light_wvfms, axis= -1)

        return None
    
    def plot_wvfms(self, event, adc, chan, xlim=None):
        '''
        Plot the light waveforms.

        Args:
            
            
        Return:
            None
        '''
        # Compute the x-axis coordinate
        x_ticks = np.arange(0, self.light_wvfms[0].shape[-1],  1)

        if (xlim==None):
            xlim = [x_ticks[0], x_ticks[-1]]


        # Setup the plot
        fig_wvfm = plt.figure(figsize=[12.8, 4.8])
        ax_wvfm = fig_wvfm.subplots()
        if (xlim != None):
            ax_wvfm.set_xlim(xlim)
        ax_wvfm.set_xlabel('ticks')
        ax_wvfm.set_ylabel('ADC unit')
        ax_wvfm.grid(True)

        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], label=f'Event {event}, ADC {adc}, Chan. {chan}', marker='.', ls='')
        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='', ls='-', c='r', alpha=0.3)

        ax_wvfm.hlines(self.baselines[event, adc, chan], xlim[0], xlim[1], color='k', ls='--', label='Baseline')

        return None
    
def _gauss(x, norm, mu, sigma):
        return norm * np.exp(-(x - mu)**2 / (2 * sigma**2))
    
def _multigauss(x, norm1, mu1, sig1, norm2, mu2, sig2, norm3, mu3, sig3):
    y = 0.
    y += _gauss(x, norm1, mu1, sig1)
    y += _gauss(x, norm2, mu2, sig2)
    y += _gauss(x, norm3, mu3, sig3)
    return y

def _group_inactive_channels(inactive_channels):
        groups_inactive_channel = []
        start_chan = inactive_channels[0]
        prev_chan = inactive_channels[0]

        for chan in inactive_channels[1:]:
            if chan == prev_chan + 1:
                # still consecutive, extend the run
                prev_chan = chan
            else:
                # break and save the [ststart_chanart, prev_chan] range
                groups_inactive_channel.append([start_chan, prev_chan])
                start_chan = chan
                prev_chan = chan
        groups_inactive_channel.append([start_chan, prev_chan])  # add the last range
        return groups_inactive_channel

