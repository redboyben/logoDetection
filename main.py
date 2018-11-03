# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 22:51:11 2018

@author: Benoit

This script requires tensorflow, opencv and numpy; it has been tested on python 3.6.
It is pretty straightforward, which is why no objects where used - although it totally could have been added.

"""
import argparse
import cv2
from detectionFunctions import loadDetectionModel, treatVideo
import csv

def touchOutputFile(output_file):
    """Create and write first row of output file
    
    Parameters
    ----------
    output_file: str
        Path to output file

    Returns
    -------
    None
        Might raise exception if bad path given
    """
    with open(output_file, 'w', newline='') as o:
        w = csv.writer(o, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['frame_id', 'x', 'y', 'w', 'h'])

def writeResults(output_file, res):
    """Create and write first row of output file
    
    Parameters
    ----------
    output_file: str
        Path to output file
    res: array, shape: [N, 5]

    Returns
    -------
    None
        File is already created
    """
    with open(output_file, 'a', newline='') as o:
        w = csv.writer(o)
        w.writerows(res) # rowS this time

def main(input_file, output_file, verbose, display_frequence):
    """Main function
    
    Parameters
    ----------
    input_file
        Path to the input file
    output_file: str
        Path to output file
    verbose: boolean
        Displaying evolution in console or not
    display_frequence: integer
        Frequence of display (seconds); not used if verbose is set to False

    Returns
    -------
    None
        The results are stored in a csv
    """
    touchOutputFile(output_file)
    res = treatVideo(input_file, verbose, display_frequence)
    writeResults(output_file, res)


if __name__ == "__main__":
    # Retrieving arguments
    parser = argparse.ArgumentParser(description = 'Detection of logos in a video')
    parser.add_argument('-i', '--input_file', type = str, nargs = 1, help = 'path/to/input_file.avi', required = True)
    parser.add_argument('-o', '--output_file', type = str, nargs = 1, help = 'path/to/output_file.csv', required = True)
    parser.add_argument('-v', '--verbose', type = bool, nargs = 1, help = 'Displaying progress or not', default = True)
    parser.add_argument('-f', '--display_frequence', type = int, nargs = 1, help = 'if verbose, how often progress is displayed', default = 15)
    args = parser.parse_args()
    main(args.input_file[0], args.output_file[0], args.verbose, args.display_frequence)
    # try:
    #     main(args.input_file, args.output_file, args.verbose, args.display_frequence)
    # except Exception as e:
    #     print("Error encountered:")
    #     print(str(e))

