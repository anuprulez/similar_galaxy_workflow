"""
Extract steps, tools and input and output types of workflows.
"""
import re
import sys
import os
import json
import time


class ExtractWorkflows:

    @classmethod
    def __init__( self ):
        """ Init method. """
        print("workflow extracted")


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python extract_workflows.py" )
        exit( 1 )
    extract_workflow = ExtractWorkflows()
