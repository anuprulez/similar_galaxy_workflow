"""
Predict tool usage to weigh the predicted tools
"""

import sys
import numpy as np
import time
import os
import warnings

import utils

warnings.filterwarnings("ignore")


class ToolPopularity:

    @classmethod
    def __init__( self ):
        """ Init method. """
        
    @classmethod
    def extract_tool_usage(self, file_path):
        """
        Extract the tool usage over time for each tool
        """
        print(file_path)

    @classmethod
    def learn_tool_popularity(self):
        """
        Fit a curve for the tool usage over time to predict future tool usage
        """
        print("Learn tool popularity")
   

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict_tool_usage.py <tool_usage_file>")
        exit( 1 )
    start_time = time.time()
    
    tool_usage = ToolPopularity()
    tool_usage.extract_tool_usage(sys.argv[1])
    
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
