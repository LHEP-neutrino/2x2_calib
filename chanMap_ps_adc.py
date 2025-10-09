import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging
from datetime import datetime
import h5py
from calibWvfms import calibWvfms

# Setup module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level

# Set global variables
N_MODULE = 4
N_ADC = 8
N_SIPM_PS = 2
SIPM_PS_NAME = ['01', '23']

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
        logger.error(f"Invalid log level: {level_name}. Should be DEBUG, INFO, WARNING, ERROR or CRITICAL")


def resolve_google_sheets_url(url_or_path):
    if url_or_path.startswith("http") and "docs.google.com/spreadsheets" in url_or_path:
        if "/d/" in url_or_path:
            file_id = url_or_path.split("/d/")[1].split("/")[0]
            # Default to first sheet
            export_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv"
            # Check if gid is in the URL (specific tab)
            if "gid=" in url_or_path:
                gid = url_or_path.split("gid=")[1].split("&")[0]
                export_url += f"&gid={gid}"
            return export_url
    return url_or_path


def get_run_info(run_tracking_file):
    # Handle Google Sheets URL
    run_tracking_file = resolve_google_sheets_url(run_tracking_file)
    logger.info(f"Run tracking file found: {run_tracking_file}")

    # Load Run tracking file as CSV, drop empty cells
    df = pd.read_csv(run_tracking_file, header=1).dropna(how="all", axis=1).dropna(how="all", axis=0)
    print(df.head(10))
    
    # Check that we have the desired info (columns)
    if "PS_chan_on" not in df.columns or "File_Name" not in df.columns:
        raise ValueError("Run tracking file must contain 'PS_chan_on' and 'File_Name' columns.")
    
    # Build the mapping dict with filename (without extension) as key
    mapping = {row["File_Name"].split(".")[0]: {"ps_chan": row["PS_chan_on"]}
           for _, row in df.iterrows()
           if pd.notna(row["PS_chan_on"])}
    logger.info(f"Loaded {len(mapping)} entries from run tracking file")
    return mapping 


def process_data_files(input_folder, mapping, pdf_path):
    
    with PdfPages(pdf_path) as pdf:
        for file_name in mapping.keys():

            # Redefine file_name for debugging
            file_name = 'mpd_run_calib_rctl_1318'

            file_path = os.path.join(input_folder, file_name+".FLOW.hdf5")
            print(file_path)
            if not os.path.exists(file_path):
                logger.warning(f"File {file_name} not found in input folder, skipping")
                continue

        
            logger.info(f"Processing file: {file_name}")

            # Use the calibWvfms class to plot the waveforms
            calibWvfm = calibWvfms(filedir=input_folder, filename=file_name+".FLOW.hdf5")
            
            try:
                # Open files
                f = h5py.File(file_path, 'r')
                
                # Load light waveform datasets
                light_wvfms = f['light/wvfm/data']['samples']

                N_adc_module = int(N_ADC/N_MODULE)
                for n_module in range(N_MODULE):

                    # -------- TODO: Part to replace with your code to find the max waveforms --------

                    # Get the max (flatten index) per PS sipm board (== module)
                    flat_index = np.argmax(light_wvfms[:,n_module*N_adc_module:(n_module+1)*N_adc_module,:,:])
                    # print(f"adc: {n_module*N_adc_module} to {(n_module+1)*N_adc_module}")

                    # Retrieve the indices of the max value
                    max_event, max_adc, max_chan, max_pt = np.unravel_index(flat_index, light_wvfms[:,n_module*N_adc_module:(n_module+1)*N_adc_module,:,:].shape)

                    # -------------------------------------------------------------------------------

                    # Save it in the mapping
                    mapping[file_name][f"mod{n_module}"] = {'adc' : int(max_adc), 'adc_chan' : int(max_chan), 'max_event' : int(max_event)} 

                    # print(f"indices: {max_event, max_adc, max_chan, max_pt}")
                    # print(f"max point: {light_wvfms[max_event, max_adc, max_chan, max_pt]}")

                    # Plot the waveform and save it in the pdf
                    calibWvfm.plot_wvfm(event=max_event, adc=max_adc, chan=max_chan, xlim=None, verbose=False, baseline=None, show_plot=False, output=pdf, peakFinder=False, minWidth=5, cut_offset=0, min_peak_distance=None)
                    
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")

            print(mapping[file_name])
            # Return after the first processed file for debug
            return 0

    return mapping


def main(input_folder, run_tracking_file, output_folder=None, log_level=None):
    if run_tracking_file is None:
        # By default: cold commissining summer 25
        run_tracking_file = 'https://docs.google.com/spreadsheets/d/108_KW4N0X-toa_DJ4d_fMkeGLiYNEdT4uE1bGmwNDY4/edit?gid=344844101' 

    # Set output folder
    if output_folder is None:
        output_folder = os.getcwd()
    os.makedirs(output_folder, exist_ok=True)

    # Set logger level
    if log_level is not None:
        set_log_level(log_level)

    # Define output files name
    today_str = datetime.now().strftime("%Y%m%d")
    pdf_path = os.path.join(output_folder, f"{today_str}_chanMap_PS_ADC_wvfms.pdf")
    csv_path = os.path.join(output_folder, f"{today_str}_chanMap_PS_ADC.csv")
    
    # Get the mapping file_name - PS_chan from the run tracking file
    mapping = get_run_info(run_tracking_file)
    
    # Get the mapping file_name - ADC_chan
    mapping = process_data_files(input_folder, mapping, pdf_path)

    # Temp. return to only test process_data_files
    return 0
    
    # ------------- TODO: write a csv file from mapping dict ----------------
    # This part of the code was not tested yet

    # Save mapping as CSV sorted by PS_chan_on
    mapping_df = pd.DataFrame.from_dict(mapping, orient="index")
    mapping_df.index.name = "PS_chan_on"
    mapping_df.sort_index(inplace=True)
    mapping_df.to_csv(csv_path)

    logger.info(f"Saved event summary to {pdf_path}")
    logger.info(f"Saved mapping to {csv_path}")
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Process data files to map the power supply channel number to the ADC channel number")
    parser.add_argument("input_folder", help="Folder containing input data files")
    parser.add_argument("--run_tracking_file", default=None, help="CSV file or Google Sheets URL containing the run description")
    parser.add_argument("--output_folder", default=None, help="Folder to save results")
    parser.add_argument("--log_level", default=None, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Level of the logger")
    
    args = parser.parse_args()

    # Run the main function
    main(args.input_folder, args.run_tracking_file, args.output_folder, args.log_level)
