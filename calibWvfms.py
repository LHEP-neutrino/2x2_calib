import sys
import subprocess
import numpy as np
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
import logging
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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
        Class to set up a calibration routine

        Inputs to this class are as follows:

            - filedir          (str):   Path to input file
            - filename         (str):   Name of input flow file
            - ouput_path       (str):   Path where to save the figures, if None: saved in ./calibWvfms_{filename} (default: None)
            - log_level        (str):   Log level
                    - 'DEBUG'
                    - 'INFO'
                    - 'WARNING'
                    - 'ERROR'

        Class methods:

            - findPeak_wvfms:           Find the peak in a waveform 
            - plot_wvfm:                Plot a single waveform
            - plot_wvfms:               Plot a set of waveforms
            
    '''

    # Initialize the class
    def __init__(self, filedir, filename, output_path=None, log_level=None):

        # Set the log level
        if log_level != None:
            set_log_level(log_level)
        
        # Open files
        f = h5py.File(os.path.join(filedir, filename), 'r')
 
        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'calibWvfms_{self.filename.split(".")[0]}/')
        else:
            self.output_path = os.path.abspath(output_path)

        # Load light events, waveform datasets
        self.light_wvfms = f['light/wvfm/data']['samples']

        self.Nevents, self.Nadc, self.Nchan, self.Ntick = self.light_wvfms.shape

        # Initialize variable(s) 
        self.Npeaks = np.zeros(self.light_wvfms.shape[:-1])

        
        # Defined some module properties
        self.N_sipm_side = int(24)
        self.N_mod = 4
        self.N_side_tpc = 2
        self.N_tpc_mod = 2
        self.N_sipm_lightModule = 6
        self.N_LCM_lightModule = 3
        self.time_tick = 16*10**-9 # [s]

        # Fingerplot variables
        self.extrem_Int_val = [[None for _ in range(self.Nchan)] for _ in range(self.Nadc)] #list of event number that have some strange int values
        self.fingerplots = [[None for _ in range(self.Nchan)] for _ in range(self.Nadc)]


        # Fit variables
        self.fit_lim = np.array([[np.array([0, self.Ntick-1], dtype=int) for _ in range(self.Nchan)] 
                        for _ in range(self.Nadc)])
        self.fitted_params = np.empty((self.Nadc, self.Nchan), dtype=object)
        self.fitted_pcov = np.empty((self.Nadc, self.Nchan), dtype=object)
        self.reduced_chi_squared = np.array([[np.array([0, 0], dtype=float) for _ in range(self.Nchan)] 
                        for _ in range(self.Nadc)])
        self.fit_status = np.zeros((self.Nadc, self.Nchan), dtype=int)


        # Gain variables
        self.gains = np.empty((self.Nadc, self.Nchan))
        self.gains_std = np.empty_like(self.gains)



        logger.info(f'Processing file {filedir+filename}')
        logger.info(f'The output path is set to {self.output_path}')
        logger.info(f"Number of events in the selection: {self.Nevents}")
    
    def load_file(self, filedir, filename):
        # Open files
        f = h5py.File(os.path.join(filedir, filename), 'r')
 
        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename

        # Load light events, waveform datasets
        self.light_wvfms = f['light/wvfm/data']['samples']

        self.Nevents, self.Nadc, self.Nchan, self.Ntick = self.light_wvfms.shape

        logger.info(f'A new file was loaded: {filedir+filename}')
        logger.info(f"Number of events in the file: {self.Nevents}")

        return None


    
    def findPeak_wvfms(self, event, adc, chan, minWidth=5, verbose=False, cut_offset=0, min_peak_distance=None):
        '''
        Find the number of peak and store it in self.Npeaks

        Args:
            
            
        Return:
            None
        '''

        peak_idx, *peak_info = _extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose, cut_offset, min_peak_distance)
        self.Npeaks[event][adc,chan] = len(peak_idx)
        if (verbose==True):
            print(f'{len(peak_idx)} peaks were found with a minWidth of {minWidth}.')
        
        self.Npeaks[event][adc,chan]=len(peak_idx)

        return None


    def plot_wvfm(self, event, adc, chan, xlim=None, verbose=False, baseline=None, show_plot=False, output=None, peakFinder=False, minWidth=5, cut_offset=0, min_peak_distance=None, integral=None):
        '''
        Plot the light waveform of adc <adc>, channel <chan> for the event <event>. Optionally the peak finder can be added to the plot

        Args:
            - event                     (int): Event number
            - adc                       (int): ADC number
            - chan                      (int): Channel number
            - xlim        (tuple, Array-Like): X-axis range
            - verbose                  (bool): Inculde additional information
            - baseline                (float): Baseline value to include on the plot
            - show_plot                (bool): If 'False' the matplotlib object is closed to avoid behind displayed in a jupyter notebook
            - output                         : Save the figure
                type:   * PdfPages : Save the figure in the pdf
                        * str      : Save the figure in the given folder with the given name (e.g. /figure/folder/wvfm.png)

            - peakFinder               (bool): Include peakFinder on the plot
            - minWidth                  (int): Minimum width of the peak
            - cut_offset              (float): Offset of the peak finder cut (peak > mean(wvfm) + cut_offset)
            - min_peak_distance         (int): Minimal distance between two peaks
            - integral                 (list): Print out the integral bounds [xlim[0], xlim[1]] and the value in the legend
           
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

        if (peakFinder==True):
            peak_idx, *peak_info = _extract_peak(self.light_wvfms[event][adc,chan], minWidth, verbose, cut_offset=cut_offset,min_peak_distance=min_peak_distance)
            mean = peak_info[0]
            # if len(peak_idx) < 2:
            #     plt.close()
            #     return None

            if (verbose==True):
                

                if cut_offset>0:
                    ax_wvfm.hlines(mean+cut_offset, xlim[0], xlim[1], color='r', ls='--', label=f'Mean+{cut_offset}')
                else:
                    ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')

                threshold = mean + cut_offset
                aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > threshold)[0]]
                belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= threshold)[0]]
                ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='', label='Accepted peaks')
                ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='', label='Discarded peaks')    

                self.Npeaks[event][adc,chan]=len(aboveMean_idx)

                print(f'{len(aboveMean_idx)} peaks were found above the threshold with a minWidth of {minWidth} and {len(belowMean_idx)} were cut out.')

            else:
                ax_wvfm.plot(peak_idx, self.light_wvfms[event][adc,chan][peak_idx], color='g', marker='x', ls='', label=f'Peaks found ({len(peak_idx)})')
                if cut_offset>0:
                    ax_wvfm.hlines(mean+cut_offset, xlim[0], xlim[1], color='r', ls='--', label=f'Mean+{cut_offset}')
                else:
                    ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')
                self.Npeaks[event][adc,chan]=len(peak_idx)



        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='.', ls='', label=f' Data points')# Event {event}, ADC {adc}, Chan. {chan}')
        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='', ls='-', c='r', alpha=0.3)
        
        if (baseline is not None):
            ax_wvfm.hlines(baseline, xlim[0], xlim[1], color='k', ls='--', label='Baseline')

        if (integral is not None):
            int = np.sum(self.light_wvfms[event][adc,chan][integral[0]:integral[1]])
            ax_wvfm.vlines(integral, ymin=ax_wvfm.get_ylim()[0], ymax=ax_wvfm.get_ylim()[1], color='C1', ls='-', label=f'Integral = {int}', zorder=10)

        ax_wvfm.legend()
        ax_wvfm.set_title(f'Waveforms of Event {event} for ADC {adc}, Chan. {chan}')

        if isinstance(output, PdfPages):
            output.savefig()
            plt.close()
        elif isinstance(output, str):
            fig_wvfm.savefig(output)
        elif output is not None: 
            logger.error("Invalid 'output' input, should be None, str or PdfPages")
        if show_plot == False:
            plt.close()

        return None
    
    def plot_wvfms(self, events=None, adcs=None, chans=None, xlim=None, verbose=False, baseline=None, show_plots=False, save_plots=False, peakFinder=False, minWidth=5, cut_offset=0, min_peak_distance=None):
        '''
        Plot the light waveforms of the given parameters. Optionally the peak finder can be added to the plot.

        Args:
            - events           : Event(s) number
                type:   * None  : all the events
                        * int   : event 'events'
                        * list  : list of events to plot
            - adcs             : ADC(s) number
                type:   * None : all adcs
                        * int, ArrayLike : the given adc numbers
            - chans            : Channel(s) number
                type:   * None : all channels
                        * ArrayLike : the given channel numbers
            - xlim        (tuple, Array-Like): X-axis range
            - verbose                  (bool): Inculde additional information
            - baseline                (float): Baseline value to include on the plot
            - show_plot                (bool): If 'False' the matplotlib object is closed to avoid behind displayed in a jupyter notebook
            - save_plots               (bool): Save the plots 
            - peakFinder               (bool): Include peakFinder on the plot
            - minWidth                  (int): Minimum width of the peak
            - cut_offset              (float): Offset of the peak finder cut (peak > mean(wvfm) + cut_offset)
            - min_peak_distance         (int): Minimal distance between two peaks
           
        Return:
            None
        '''
        
        
        inputFile_name = self.filename.split(".")[0]

        if events is None:
            events = np.arange(0, self.Nevents)
        else:
            if isinstance(events, int):
                events = np.array([events])
            elif (isinstance(events, list)):
                events = np.array(events)
            else:
                logger.error("Invalid 'events' input, should be None int or list")
        
        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        elif isinstance(adcs, int) or isinstance(adcs, np.int64):
            adcs = np.array([adcs])
        elif (isinstance(adcs, list)):
            adcs = np.array(adcs)
        else:
            logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, self.Nchan)
        elif isinstance(chans, int) or isinstance(chans, np.int64):
            chans = np.array([chans])
        elif (isinstance(chans, list)):
            chans = np.array(chans)
        else:
            logger.error("Invalid 'chans' input, should be None, int or list")
        print(events, adcs, chans)

        for i_event in events:
            for j_adc in adcs:
                    for k_chan in chans:
                        if save_plots == False:
                            output_plot = None
                        elif save_plots == True:
                            output_path = os.path.join(self.output_path, inputFile_name, f'adc{j_adc}', f'chan{k_chan}')
                            os.makedirs(output_path, exist_ok=True)
                            output_plot=f"{output_path}/Ev{i_event}_adc{j_adc}_chan{k_chan}_wvfm.png"

                        self.plot_wvfm(i_event, j_adc, k_chan, xlim=xlim, peakFinder=peakFinder, minWidth=minWidth, baseline=baseline, verbose=verbose, output=output_plot, show_plot=show_plots, cut_offset=cut_offset, min_peak_distance=min_peak_distance)
    
        return None
    

    
    def _plot_summaryDC(self, adc, show_plot=False, output=None, peakFinder_minWidth=5, inactive_channels=None, cut_offset=0, min_peak_distance=None):
        # Compute the x-axis coordinate
        x_chan = np.arange(0, self.Nchan,  1)
        xlim = [x_chan[0]-0.5, x_chan[-1]+0.5]

        if inactive_channels is not None:
            mask_inactive_chans = np.isin(x_chan, inactive_channels)
            x_chan = x_chan[~mask_inactive_chans]

        Nevents = 100 # self.Nevents

        # Setup the plot
        fig_summaryDC = plt.figure(figsize=[12.8, 4.8])
        ax_summaryDC = fig_summaryDC.subplots()
        
        ax_summaryDC.set_xlim(xlim)
        ax_summaryDC.set_xticks(np.arange(xlim[0]+0.5, xlim[-1]-0.5, 2))
        
        ax_summaryDC.set_xlabel('Channel')
        ax_summaryDC.set_ylabel('mean DC rate [kHz]')
        ax_summaryDC.grid(True)

        # Compute DC rate
        for i_event in range(Nevents):
            for j_chan in x_chan:
                self.findPeak_wvfms(i_event, adc, j_chan, minWidth=peakFinder_minWidth, cut_offset=cut_offset, min_peak_distance=min_peak_distance)

        peaks_sum = np.sum(self.Npeaks[:Nevents,adc,x_chan], axis=0)
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

    
    def plot_summaryDC(self, events=None, adcs=None, chans=None, show_plots=False, save_plots=False, peakFinder_minWidth=5, inactive_channels=None, cut_offset=0, min_peak_distance=None):
        '''
        Plot the DC rate in function of channel numbers, don't include the inactive channels 'inactive_channels'

        Args:
            - events           : Event(s) number
                type:   * None : all the events
                        * int : event 'events'
                        * list, tuple: from event 'events[0]' to 'events[1]'
            - adcs             : ADC(s) number
                type:   * None : all adcs
                        * int, ArrayLike : the given adc numbers
            - chans            : Channel(s) number
                type:   * None : all channels
                        * ArrayLike : the given channel numbers
            - show_plot                (bool): If 'False' the matplotlib object is closed to avoid behind displayed in a jupyter notebook
            - save_plots               (bool): Save the plot in the output folder
            - peakFinder_minWidth       (int): Minimum width of the peak for the peakFinder
            - inactive_channels       (list) : List of inactive channels, left out of computation
            - cut_offset              (float): Offset of the peak finder cut (peak > mean(wvfm) + cut_offset)
            - min_peak_distance         (int): Minimal distance between two peaks
           
        Return:
            None
        '''
        
        inputFile_name = self.filename.split(".")[0]

        if events is None:
            events = np.array([0, self.Nevents])
        else:
            if isinstance(events, int):
                events = np.array([events, events+1])
            elif (isinstance(events, (list, tuple)) and len(events) == 2):
                events = np.array(events)
            else:
                logger.error("Invalid 'events' input, should be None, int, list or tuple")

        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or ArrayLike")


        if chans is None:
            chans = np.array([0, self.Nchan])
            
        for j_adc in adcs:
            if save_plots == False:
                output_plot = None
            else:
                output_path = os.path.join(self.output_path, inputFile_name, f'adc{j_adc}')
                os.makedirs(output_path, exist_ok=True)
                output_plot=f"{output_path}/summaryDC_adc{j_adc}.png"
            
            self._plot_summaryDC(j_adc, show_plot=show_plots, output=output_plot, peakFinder_minWidth=peakFinder_minWidth, inactive_channels=inactive_channels, cut_offset=cut_offset, min_peak_distance=min_peak_distance)

        return None
    
    def plot_DC_pdf(self, event=7, adcs=None, chans=None, peakFinder_minWidth=5, inactive_channels=None, cut_offset=0, min_peak_distance=None):
        '''
        Regroup in a pdf the DC rate in function of channel numbers (inactive channels 'inactive_channels' not included) 
        and a waveform  example of each channel

        Args:
            - event          (int) : Event number for the example waveforms
            - adcs             : ADC(s) number
                type:   * None : all adcs
                        * int, list : the given adc numbers
            - chans            : Channel(s) number
                type:   * None : all channels
                        * list : the given channel numbers
            - peakFinder_minWidth       (int): Minimum width of the peak for the peakFinder
            - inactive_channels       (list) : List of inactive channels, left out of computation
            - cut_offset              (float): Offset of the peak finder cut (peak > mean(wvfm) + cut_offset)
            - min_peak_distance         (int): Minimal distance between two peaks
           
        Return:
            None
        '''
        
        inputFile_name = self.filename.split(".")[0]

        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, self.Nchan)
        elif (isinstance(chans, list)):
            chans = np.array(chans)
        else:
            logger.error("Invalid 'chans' input, should be None or list")

        output_pdf = os.path.join(self.output_path, inputFile_name)
        os.makedirs(output_pdf, exist_ok=True)

        output_path = os.path.join(self.output_path, inputFile_name)
        logger.info(f'The summary pdf will be created in the folder {output_path}')

        for j_adc in adcs:
            output_path_adc = os.path.join(output_path, f'adc{j_adc}')
            os.makedirs(output_path_adc, exist_ok=True)
            output_pdf=f"{output_path_adc}/summaryDC_adc{j_adc}.pdf"

            with PdfPages(output_pdf) as summary_pdf:
                self._plot_summaryDC(j_adc, show_plot=False, output=summary_pdf, peakFinder_minWidth=peakFinder_minWidth, inactive_channels=inactive_channels, cut_offset=cut_offset, min_peak_distance=min_peak_distance)
                for k_chan in chans:
                    self.plot_wvfm(event, j_adc, k_chan, peakFinder=True, minWidth=peakFinder_minWidth, output=summary_pdf, show_plot=False, cut_offset=cut_offset, min_peak_distance=min_peak_distance)
    
            logger.info(f'The summary pdf of adc {j_adc} was created')
        
        return None
    
    def set_output_path(self, output_path):
        self.output_path = os.path.abspath(output_path)
        logger.info(f'The output path was updated to {self.output_path}')
        return None
    
    def compute_fingerplots(self, Nevent=None, adcs=None, chans=None, int_window=[0, -1], Nbins=150, mode='integral', cut=None, nSig=5):
        '''
        Plot the distribution of integrated waveforms, so called fingers plot

        Args:
            Nevent (int)                    : Number of event include in the computation (default: -1, all events)
            int_window (np.array or list)   : Integaration window, the waveform will be integrated from int_window[0]
                                              to int_window[1]
            mode (str)                      : Mode of computation
                - 'integral' (default) : Integrate the waveform in the given window
                - 'amplitude'          : Maximal amplitude of the peak
                - 'amplitude_peaks'    : Includes all the peaks found by the peak finder

                - 'fit_int'            : Integral of the fitted waveform, TODO
                - 'fit_amp'            : Max. amplitude of the fitted waveform, TODO
            cut (str)                       : Cut(s) applied to the select event
                - None (default)       : No cut
                - 1peak                : Only select event with one peak in 'int_window' 
                - 15ticks              : Only select peak at least 15 ticks away from neighbouring peaks   
        '''

        # Cut variable
        minWidth = 9
        logger.debug(f"Computing the fingerplots of the file {self.filename} with {Nevent} events")

        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, self.Nchan)
        elif (isinstance(chans, list)):
            chans = np.array(chans)
        elif (isinstance(chans, int)):
            chans = np.array([chans])
        else:
            logger.error("Invalid 'chans' input, should be None, int or list")

        if Nevent is None or Nevent > self.light_wvfms.shape[0]:
            Nevent = self.light_wvfms.shape[0]

        
        logger.debug(f"ADCs: {adcs}")
        logger.debug(f"Chans: {chans}")

        
        if (mode == 'integral'):
            for i_adc in adcs:
                for j_chan in chans:
                    if (cut == '1peak'):
                        self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window)
                        event_mask = np.full((Nevent), True, dtype=bool)
                        for k_event in range(Nevent):
                            event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
                        event_mask = np.where(event_mask)[0]
                        intsWvfm = np.sum(self.light_wvfms[event_mask, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)

                    else:
                        intsWvfm = np.sum(self.light_wvfms[:Nevent+100, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)
                        print(f"shape: {intsWvfm.shape}")#, values: {intsWvfm}")
                        Int_extremValues = check_for_extrem_values(intsWvfm, nSig=nSig)
                        self.extrem_Int_val[i_adc][j_chan] = Int_extremValues
                        # Delete the extrem values
                        intsWvfm = np.delete(intsWvfm,Int_extremValues)

                    if len(intsWvfm)<Nevent:
                        Nevent = len(intsWvfm)
                    
                    self.fingerplots[i_adc][j_chan] = np.histogram(intsWvfm[:Nevent], bins=Nbins)

        # elif (mode == 'amplitude'):
        #     for i_adc in range(self.Nadc):
        #         for j_chan in range(self.Nchan):
        #             if (cut == '1peak'):
        #                 self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window)
        #                 event_mask = np.full((Nevent), True, dtype=bool)
        #                 for k_event in range(Nevent):
        #                     event_mask[k_event] = (len(self.peaks_idx[k_event][i_adc][j_chan])==1)
        #                 event_mask = np.where(event_mask)[0]
        #                 ampWvfm = np.max(self.light_wvfms[event_mask, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)


        #             else:
        #                 ampWvfm = np.max(self.light_wvfms[:Nevent, i_adc, j_chan, int_window[0]:int_window[1]], axis=-1)

        #             self.fingerplots[i_adc][j_chan] = np.histogram(ampWvfm, bins=Nbins)

        # elif (mode == 'amplitude_peaks'):
        #     for i_adc in range(self.Nadc):
        #         for j_chan in range(self.Nchan):
        #             self.findPeak_wvfms([0, Nevent], i_adc, j_chan, minWidth=minWidth, search_int=int_window, cut=cut)

        #             ampsWvfm = np.array([])
        #             for k_event in range(Nevent):
        #                 ampsWvfm = np.concatenate((ampsWvfm, self.light_wvfms[k_event, i_adc, j_chan, 
        #                                             self.peaks_idx[k_event][i_adc][j_chan]]), axis=None)

        #             self.fingerplots[i_adc][j_chan] = np.histogram(ampsWvfm, bins=Nbins)

        logger.info(f'The finger plots were computed with {Nevent} events in mode "{mode}"')

        return None
    
    def _compute_fingerplots_p0(self, counts, bin_centers, width, posRatio_noisePeak = 0.4):
        '''
        Compute the initial parameter 'p0' for  the fingerplots fit and the bounds of the fit parameters.

        Args:
            counts (np.array):      The values of the histogram (returned from np.histogram)
            bin_centers (np.array): Center of the bins corresponding to 'counts'
            width (int):            Width of the bins w.r.t. counts and bin_centers


        Return:
            fit_p0 (np.array):       The initial parameters for the fingerplot fit
            fit_bounds (np.array):   The bounds of the fit parameters
            fit_lim (np.array):      An array containing the range where the fit will be computed
        '''
        # Get the peaks position above <height>
        peaks, properties = find_peaks(counts, distance=5, width=1, prominence=20, height=np.mean(counts))

        Npeaks_fit = len(peaks)

        # Add a second peak for each main peak, representing the an observed noise effect (cross-talk?)
        # Compute the noise peak position w.r.t. the distance between the main peaks
        noise_peaks = np.array((peaks[1:]-peaks[:-1])*posRatio_noisePeak, dtype=int)
        # Add the noise peak of the last main peak
        noise_peaks = np.append(noise_peaks, noise_peaks[-1])
        # Get the absolute position
        noise_peaks += peaks

        # Get the p0 and the fit bounds
        # 2 gaussian per peak and 3 params per gaussian
        self.Nparams_peak = 6
        fit_p0 = np.zeros(Npeaks_fit*self.Nparams_peak)
        fit_bounds = np.zeros((2,Npeaks_fit*self.Nparams_peak))
        
        for i_peak in range(Npeaks_fit):
            # Amplitue main gaussian
            fit_p0[i_peak*self.Nparams_peak] = counts[peaks[i_peak]]
            fit_bounds[0][i_peak*self.Nparams_peak] = fit_p0[i_peak*self.Nparams_peak]*0.95
            fit_bounds[1][i_peak*self.Nparams_peak] = fit_p0[i_peak*self.Nparams_peak]*1.1+1 # +1 to avoid both bound being 0, if fit_p0[...] is 0
            # Mean main gaussian
            fit_p0[i_peak*self.Nparams_peak+1] = bin_centers[0]+width*peaks[i_peak]
            fit_bounds[0][i_peak*self.Nparams_peak+1] = fit_p0[i_peak*self.Nparams_peak+1]-width
            fit_bounds[1][i_peak*self.Nparams_peak+1] = fit_p0[i_peak*self.Nparams_peak+1]+width
            # Std main gaussian
            fit_p0[i_peak*self.Nparams_peak+2] = properties['widths'][i_peak]*width*0.5
            fit_bounds[0][i_peak*self.Nparams_peak+2] = fit_p0[i_peak*self.Nparams_peak+2]*0.8
            fit_bounds[1][i_peak*self.Nparams_peak+2] = fit_p0[i_peak*self.Nparams_peak+2]*2+1 # +1 to avoid both bound being 0, if fit_p0[...] is 0
            # Amplitue secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+3] = counts[noise_peaks[i_peak]]
            fit_bounds[0][i_peak*self.Nparams_peak+3] = float(fit_p0[i_peak*self.Nparams_peak+3])*0.6
            fit_bounds[1][i_peak*self.Nparams_peak+3] = float(fit_p0[i_peak*self.Nparams_peak+3])*1.1+1 # +1 to avoid both bound being 0, if fit_p0[...] is 0
            # Mean secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+4] = bin_centers[0]+width*noise_peaks[i_peak]
            fit_bounds[0][i_peak*self.Nparams_peak+4] = fit_p0[i_peak*self.Nparams_peak+4]-width
            fit_bounds[1][i_peak*self.Nparams_peak+4] = fit_p0[i_peak*self.Nparams_peak+4]+2*width
            # Amplitue secondary gaussian
            # Std secondary gaussian
            fit_p0[i_peak*self.Nparams_peak+5] = (properties['widths'][i_peak]*width*0.5)
            fit_bounds[0][i_peak*self.Nparams_peak+5] = fit_p0[i_peak*self.Nparams_peak+5]*0.8
            fit_bounds[1][i_peak*self.Nparams_peak+5] = fit_p0[i_peak*self.Nparams_peak+5]*5+1 # +1 to avoid both bound being 0, if fit_p0[...] is 0

            # logger.debug(f" prop: {properties['left_ips'][0]}")
            if properties['left_ips'][0] > 0 and counts[int(properties['left_ips'][0]-1)]>0:
                    fit_lim_min = properties['left_ips'][0]-1
            else:
                fit_lim_min = properties['left_ips'][0]

            if peaks[-1]+int((peaks[-1]-peaks[-2])*0.5)+1 < len(counts):
                fit_lim_max = peaks[-1]+int((peaks[-1]-peaks[-2])*0.5)+1
            else: 
                fit_lim_max = len(counts)


        return fit_p0, fit_bounds, np.array([fit_lim_min, fit_lim_max], dtype=int)
    
    def fit_fingerplots(self, adcs=None, chans=None, show_p0_plots=False, output=None):
        '''
        Fit the fingerplots. If the fit fails, saved the p0 parameters instead.

        Args:
            
        '''
        logger.debug(f"Fitting the fingerplots of the file {self.filename}")
        
        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, self.Nchan)
        elif isinstance(chans, int):
            chans = np.array([chans])
        elif (isinstance(chans, list)):
            chans = np.array(chans)
        else:
            logger.error("Invalid 'chans' input, should be None, int or list")

        logger.debug(f"ADCs: {adcs}")
        logger.debug(f"Chans: {chans}")

        for i_adc in adcs:
            for j_chan in chans:

                counts, bins = self.fingerplots[i_adc][j_chan]

                bin_centers = (bins[:-1] + bins[1:]) / 2
                width = bins[1] - bins[0]
                
                # Compute p0
                logger.debug(f"Compute fingerplots p0 of ADC {i_adc}, chan. {j_chan}")
                fit_p0, fit_bounds, fit_lim = self._compute_fingerplots_p0(counts=counts, bin_centers=bin_centers,
                                                                            width=width)
                print(f"fit_bounds: {fit_bounds}")
                print(f"fit_p0: {fit_p0}")
                self.fit_lim[i_adc][j_chan] = fit_lim
                if (show_p0_plots == True):
                    plot_fingerplot(counts, bins, title=f'Fingerplot and p0 for ADC {i_adc}, chan. {j_chan}', show_plot=True, fit_params=fit_p0, fit_xlim=self.fit_lim[i_adc][j_chan], output=output)
                if (output is not None):    
                    plot_fingerplot(counts, bins, title=f'Fingerplot and p0 for ADC {i_adc}, chan. {j_chan}', fit_params=fit_p0, fit_xlim=self.fit_lim[i_adc][j_chan], output=output)

                # Fit the fingerplots
                try:
                    
                    sigma = np.sqrt(counts[fit_lim[0]:fit_lim[1]])
                    sigma[sigma == 0] = 1 # To avoid division by zero
                    print(f"sigma: {sigma}")
                    self.fitted_params[i_adc][j_chan], self.fitted_pcov[i_adc][j_chan] = curve_fit(multi_gaussian,
                                                   bin_centers[fit_lim[0]:fit_lim[1]], 
                                                   counts[fit_lim[0]:fit_lim[1]], p0=fit_p0, 
                                                   bounds=fit_bounds, maxfev = 100000)#, sigma=sigma)
                    chi_squared = np.sum(((counts[fit_lim[0]:fit_lim[1]] - multi_gaussian(bin_centers[fit_lim[0]:fit_lim[1]], *self.fitted_params[i_adc][j_chan]))**2 / sigma))
                    dof = len(counts[fit_lim[0]:fit_lim[1]]) - len(self.fitted_params[i_adc][j_chan])  # degrees of freedom
                    # print(f'dof: {len(counts[fit_lim[0]:fit_lim[1]])} and {len(self.fitted_params[i_adc][j_chan])}, fit_lim: {fit_lim}')
                    logger.debug(f"Fingerplots of ADC {i_adc}, chan. {j_chan} fitted")
                    self.fit_status[i_adc][j_chan] = 0
                except Exception as e:
                    logger.warning(f"The fingerplot were not fitted for ADC {i_adc}, chan. {j_chan}, the saved parmaters are the p0s: {e}")
                    self.fitted_params[i_adc][j_chan] = fit_p0
                    chi_squared = 0
                    dof = 0
                    self.fit_status[i_adc][j_chan] = 1
                    # print(f"status: {self.fit_status[i_adc][j_chan]}")

                self.reduced_chi_squared[i_adc][j_chan] = np.array([chi_squared, dof])

        logger.info(f"The fingerplots of the file {self.filename} were fitted.")
        

        return None
        
    def compute_gains(self, adcs=None, chans=None, mode='integral'):
        if adcs is None:
            adcs = np.arange(0, self.Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, self.Nchan)
        elif isinstance(chans, int):
            chans = np.array([chans])
        elif (isinstance(chans, list)):
            chans = np.array(chans)
        else:
            logger.error("Invalid 'chans' input, should be None, int or list")


        for i_adc in adcs:
            for j_chan in chans:
                if self.fit_status[i_adc][j_chan] == 0:
                    Npeaks_fit = int(len(self.fitted_params[i_adc][j_chan])/self.Nparams_peak)
                    peak_diffs = np.array([self.fitted_params[i_adc][j_chan][k_peak*self.Nparams_peak+1]-self.fitted_params[i_adc][j_chan][(k_peak+1)*self.Nparams_peak+1] for k_peak in range(Npeaks_fit-1)])
                    if (mode in ['amplitude', 'amplitude_peaks']):
                        self.gains[i_adc][j_chan] = abs(np.mean(peak_diffs))
                        self.gains_std[i_adc][j_chan] = np.std(peak_diffs)
                    else : 
                        self.gains[i_adc][j_chan] = abs(np.mean(peak_diffs[1:]))
                        self.gains_std[i_adc][j_chan] = np.std(peak_diffs[1:])
                else:
                    logger.debug(f'Skipped ADC {i_adc}, chan. {j_chan} due to failed fitting')

        return None  

 # ~~~~~~~~~~~~~~~~~~ Helper funcitons ~~~~~~~~~~~~~~~~~~

def _extract_peak(wvfm, minWidth, verbose=False, cut_offset=0, min_peak_distance=None):
        ''' 
        Extract the index of the peak in a waveform. A peak is a data point where 'minWidth/2' data points
        on each side have a smaller amplitude. Additional it requires that the peak is above the mean of the
        waveform, or above the mean+cut_offset if cut_offset is not 0


        Args:
            - wvfm      (Array-Like):   1D array containing the data point of the waveform
            - minWidth         (int):   Minimal width of the peak
            - verbose         (bool):   Output more information about the peaks selected/non-selected
            - cut_offset     (float):   Log level
                    

        Return:
            - peak_idx     (ndarray):   Array containing the peak indices
            - mean           (float):   Mean of the waveform
            
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
           aboveMean_idx = np.where(wvfm[peak_idx]>mean+cut_offset)[0]
           peak_idx = peak_idx[aboveMean_idx]

        if (min_peak_distance is not None) and len(peak_idx) != 0:
            tmp_peak_idx = [peak_idx[0]]
            for peak_id in peak_idx[1:]:
                if peak_id-tmp_peak_idx[-1] > min_peak_distance :
                    tmp_peak_idx.append(peak_id)
                else:
                    if wvfm[peak_id] > wvfm[tmp_peak_idx[-1]]:
                        tmp_peak_idx[-1] = peak_id

            peak_idx = np.array(tmp_peak_idx, dtype=int)

        return peak_idx, mean 

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

def multi_gaussian(x, *params):
    """
    Compute the sum of multiple Gaussians.
    Each Gaussian has 3 parameters: amplitude, mean, std_dev.
    
    Args:
        x:          input array
        *params:    variable length parameters [A1, mu1, sigma1, A2, mu2, sigma2, ..., An, mun, sigman]
        
    Return:
        y:          sum of Gaussians evaluated at x
    """
    y = np.zeros_like(x, dtype=float)
    num_gaussians = len(params) // 3
    
    for i in range(num_gaussians):
        A = params[3*i]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]

        y += A * np.exp(-((x - mu)**2) / (2 * sigma**2))
        
    return y

def plot_fingerplot(counts, bins, title=None, show_plot=False, output=None, plot_name=None, plot_xlim=None, fit_params=None, Nparams_peak=6, fit_xlim=None, reduced_chi_squared=None, gain=None, pedestal=None, nPEs=None, fit_status=None):
        """
            Plot the finger plot.
        
        Args:
            counts (np.array):      The values of the histogram (see np.histogram docs)
            bins (np.array):        The bin edges of the histogram (see np.histogram docs)
            # mode (str):             Mode of the plot
            #     - 'default':            Plot the fingerplot 
            #     - 'fit'    :            Plot the fingerplot with the fit function passed by fit_params
            #     - 'gain' :              Plot the gain additionally to the fit function  
            # title (str):            Title of the plot, overwrite the default title
            show (bool):            Show the plot
            output :                Output, if None the plot is not saved 
                type:   * PdfPages : Save the figure in the pdf
                        * str      : Save the figure in the given folder
            plot_name:              Name of the plot, if None default name YYYYMMDD_fingerplot.png
            fit_params (np.array):  Parameters of the multi_gaussian function to plot the fitting function. Required 
                                    in mode 'fit'
            plot_xlim (list):       X-axis limits of the plot
            Nparams_peak:           Number of parameter per peak


        Return:
            None
        """
        logger.debug(f'Plotting the finger plot')

        fig = plt.figure(figsize=[10, 6])
        ax = fig.subplots()

        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]

        ax.bar(bin_centers, counts, width=width, color='skyblue', label=f'Fingerplot with {np.sum(counts)} entries', zorder= 5)

        default_title= 'Fingerplot'
        if (fit_params is not None and fit_xlim is not None):
            Npeaks = int(len(fit_params)/Nparams_peak)
            # print(f"fit_xlim: {fit_xlim}")
            x_fit = np.linspace(bin_centers[fit_xlim[0]], bin_centers[fit_xlim[1]], 1000)

            ax.plot(x_fit, multi_gaussian(x_fit, *fit_params), color='r', ls='-', label='Fitted function (multi-gaussian)', zorder= 10)
            
            default_title= 'Fingerplot with a multigaussian fit function'

            for i_peak in range(Npeaks):
                ax.plot([], [],'', label=f"Peak {i_peak + 1}: $\mu$ = {fit_params[i_peak*Nparams_peak+1]:.1f}, $\sigma$ = {fit_params[i_peak*Nparams_peak+2]:.1f}", color="None", zorder= 20)

            print(f"\n\n\nstatus: {fit_status}")   

            if fit_status == 1:
                print(f"\n\n\n in condition: status: {fit_status}")   
                ax.text(0.5, 0.5, "FAILED FIT", color='gray', fontsize=48, ha='center', va='center', alpha=0.3, zorder=10, transform=ax.transAxes)

            if (gain != None):
                if (reduced_chi_squared is not None):
                    ax.plot([], [], '', label=f"Gain: {gain:.1f}, $red. \chi^2$ = {reduced_chi_squared[0]:.1f}/{reduced_chi_squared[1]}", color="None", zorder= 30)

                else:
                    ax.plot([], [], '', label=f"Gain: {gain:.1f}", color="None", zorder= 30)

                if (nPEs is not None):
                    y_coord_text = []
                    for i_peak in range(len(nPEs)):
                        y_coord_text.append(multi_gaussian(fit_params[i_peak*Nparams_peak+1], *fit_params)+15)
                        ax.text(fit_params[i_peak*Nparams_peak+1], y_coord_text[i_peak] , f"{int(np.round(nPEs[i_peak],))} PE", 
                                fontsize=10, ha='center', va='bottom', zorder= 20)

                    ax.set_ylim([0, np.max(y_coord_text)+20])

                if (pedestal is not None):
                    ax.vlines(x=pedestal, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=f"Pedestal: {pedestal:.1f}", color="green", ls='--', zorder= 30)

            ax.vlines(x=bin_centers[fit_xlim], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color = "C1", label='Fitting bounds', zorder= 10)


            ax.add_patch(Rectangle((x_fit[0],ax.get_ylim()[0]), abs(x_fit[-1]-x_fit[0]), ax.get_ylim()[1]-ax.get_ylim()[0], color='C1', alpha=0.1, zorder= 0))

        if (title==None):
            ax.set_title(default_title, fontsize=8)
        else:
            ax.set_title(title, fontsize=8)

        if np.all(plot_xlim):
            ax.set_xlim(plot_xlim)

        handles, labels = ax.get_legend_handles_labels()
        # sort both labels and handles by labels (alphabetic order)
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels)

        ax.set_xlabel('ADC counts')
        ax.set_ylabel('Number of entries')
        ax.grid(True)

        if isinstance(output, PdfPages):
            output.savefig()
            plt.close()
        elif isinstance(output, str):
            now = datetime.now()
            time_str = now.strftime("%Y%m%d_%H%M%S")
            if (plot_name is not None):
                output_plot = os.path.join(output, f'{time_str}_{plot_name}')
            else:
                output_plot = os.path.join(output ,f'{time_str}_fingerplot.png')
        
            fig.savefig(output_plot)
            logger.debug(f'File {os.path.basename(output_plot)} saved in {os.path.dirname(output_plot)}')

        elif output is not None: 
            logger.error("Invalid 'output' input, should be None, str or PdfPages")

        if show_plot == False:
            plt.close()

        return None 

def check_for_extrem_values(Int_array, nSig = 5):
    """
        Check for extrem values in the waveforms integral values
    """
    # Compute mean and std along the 2nd dimension (int of waveforms)
    mean = np.mean(Int_array, axis=-1)
    std = np.std(Int_array, axis=-1)

    # print(f"mean: {mean}, {mean[np.newaxis]}\nstd: {std}, {std[np.newaxis]}")

    # Compute how far each value is from the mean
    # Broadcasting mean and std to match arr’s shape
    z_scores = (Int_array - mean) / std
    # print(f"z_scores: {z_scores}")

    # Boolean mask for values more than 5 standard deviations away
    mask = np.abs(z_scores) > nSig

    # Get the coordinates (i, j, k) of those outlier values
    events_extremValues = np.argwhere(mask)[:,0]
    
    return events_extremValues

