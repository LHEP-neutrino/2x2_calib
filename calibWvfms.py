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
        f = h5py.File(filedir+filename, 'r')

        # Set general class-level variables from inputs
        self.filedir = filedir
        self.filename = filename
        
        # Set the output path
        if (output_path is None):
            self.output_path = os.path.join(os.path.dirname(__file__) ,f'calibWvfms_{self.filename}/')
        else:
            self.output_path = os.path.abspath(output_path)

        # Load light events, waveform datasets
        self.light_events = f['light/events/data']
        self.light_wvfms = f['light/wvfm/data']['samples']

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

        logger.info(f'Processing file {filedir+filename}')
        logger.info(f'The output path is set to {self.output_path}')
        logger.info(f"Number of events in the selection: {len(self.light_events)}")
    

    
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


    def plot_wvfm(self, event, adc, chan, xlim=None, verbose=False, baseline=None, show_plot=False, output=None, peakFinder=False, minWidth=5, cut_offset=0, min_peak_distance=None):
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
                aboveMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] > mean)[0]]
                belowMean_idx = peak_idx[np.where(self.light_wvfms[event][adc,chan][peak_idx] <= mean)[0]]
                ax_wvfm.plot(aboveMean_idx, self.light_wvfms[event][adc,chan][aboveMean_idx], color='g', marker='x', ls='', label='Accepted peaks')
                ax_wvfm.plot(belowMean_idx, self.light_wvfms[event][adc,chan][belowMean_idx], color='r', marker='x', ls='Discarded peaks')

                if cut_offset>0:
                    ax_wvfm.hlines(mean+cut_offset, xlim[0], xlim[1], color='r', ls='--', label=f'Mean+{cut_offset}')
                else:
                    ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')

                self.Npeaks[event][adc,chan]=len(aboveMean_idx)

                print(f'{len(aboveMean_idx)} peaks were found above the mean with a minWidth of {minWidth} and {len(belowMean_idx)} were cut out.')

            else:
                ax_wvfm.plot(peak_idx, self.light_wvfms[event][adc,chan][peak_idx], color='g', marker='x', ls='', label=f'Peaks found ({len(peak_idx)})')
                if cut_offset>0:
                    ax_wvfm.hlines(mean+cut_offset, xlim[0], xlim[1], color='r', ls='--', label=f'Mean+{cut_offset}')
                else:
                    ax_wvfm.hlines(mean, xlim[0], xlim[1], color='r', ls='--', label='Mean')
                self.Npeaks[event][adc,chan]=len(peak_idx)



        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='.', ls='', label=f' Data points')# Event {event}, ADC {adc}, Chan. {chan}')
        ax_wvfm.plot(x_ticks, self.light_wvfms[event][adc,chan], marker='', ls='-', c='r', alpha=0.3)
        
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
    
    def plot_wvfms(self, events=None, adcs=None, chans=None, verbose=False, baseline=None, show_plots=False, save_plots=False, peakFinder=False, minWidth=5, cut_offset=0, min_peak_distance=None):
        '''
        Plot the light waveforms of the given parameters. Optionally the peak finder can be added to the plot.

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

        Nevents, Nadc, Nchan, _ = self.light_wvfms.shape

        if events is None:
            events = np.array([0, Nevents])
        else:
            if isinstance(events, int):
                events = np.array([events, events+1])
            elif (isinstance(events, (list, tuple)) and len(events) == 2):
                events = np.array(events)
            else:
                logger.error("Invalid 'events' input, should be None int or ArrayLike")

        if adcs is None:
            adcs = np.arange([0, Nadc])
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.array([0, Nchan])
        else:
            chans = np.array(chans)
            
        # print(events, adcs, chans)

        for i_event in range(*events):
            for j_adc in range(*adcs):
                    for k_chan in range(*chans):
                        if save_plots == False:
                            output_plot = None
                        elif save_plots == True:
                            output_path = os.path.join(self.output_path, inputFile_name, f'adc{j_adc}', f'chan{k_chan}')
                            os.makedirs(output_path, exist_ok=True)
                            output_plot=f"{output_path}/Ev{i_event}_adc{j_adc}_chan{k_chan}_wvfm.png"

                        self.plot_wvfm(i_event, j_adc, k_chan, peakFinder=peakFinder, minWidth=minWidth, baseline=baseline, verbose=verbose, output=output_plot, show_plot=show_plots, cut_offset=cut_offset, min_peak_distance=min_peak_distance)
    
        return None
    

    
    def _plot_summaryDC(self, adc, show_plot=False, output=None, peakFinder_minWidth=5, inactive_channels=None, cut_offset=0, min_peak_distance=None):
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

        Nevents, Nadc, Nchan, _ = self.light_wvfms.shape

        if events is None:
            events = np.array([0, Nevents])
        else:
            if isinstance(events, int):
                events = np.array([events, events+1])
            elif (isinstance(events, (list, tuple)) and len(events) == 2):
                events = np.array(events)
            else:
                logger.error("Invalid 'events' input, should be None, int, list or tuple")

        if adcs is None:
            adcs = np.arange(0, Nadc)
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or ArrayLike")


        if chans is None:
            chans = np.array([0, Nchan])
            
        for j_adc in adcs:
            if output == False:
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

        _, Nadc, Nchan, _ = self.light_wvfms.shape

        if adcs is None:
            adcs = np.array([0, Nadc])
        else:
            if isinstance(adcs, int):
                adcs = np.array([adcs])
            elif (isinstance(adcs, list)):
                adcs = np.array(adcs)
            else:
                logger.error("Invalid 'adcs' input, should be None, int or list")

        if chans is None:
            chans = np.arange(0, Nchan)
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

