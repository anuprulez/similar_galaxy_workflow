"""
Extract steps, tools and input and output types of workflows.
"""
import re
import sys
import os
import json
import time
import csv


class ExtractWorkflows:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.workflow_directory = 'data/workflows/'
        self.tool_data_filename = 'workflows_raw.csv'

    @classmethod
    def read_workflow_file( self, workflow_file_path, file_id ):
        """
        Read a workflow file to collect all the steps and other attributes
        """
        workflow_json = dict()
        with open( workflow_file_path, 'r' ) as workflow_file:
            file_raw = workflow_file.readline()
            try:
                file_json = json.loads( file_raw )
                workflow_json[ file_id ] = file_json
            except Exception as exception:
                pass
        return workflow_json

    @classmethod
    def read_workflow_directory( self ):
        """
        Read workflow's directory
        """
        workflow_json = list()
        for folder in os.listdir( self.workflow_directory ):
            workflow_path = os.path.join( self.workflow_directory, folder )
            for workflow_file_id in os.listdir( workflow_path ):
                wf = self.read_workflow_file( os.path.join( workflow_path, workflow_file_id ), workflow_file_id )
                if workflow_file_id not in workflow_json:
                    workflow_json.append( wf )
        print len(workflow_json)
        """with open( os.path.join( "data", self.tool_data_filename ), 'wb' ) as workflows:
            wr = csv.writer( workflows, quoting=csv.QUOTE_ALL )
            wr.writerow( workflow_json )"""


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python extract_workflows.py" )
        exit( 1 )
    start_time = time.time()
    extract_workflow = ExtractWorkflows()
    extract_workflow.read_workflow_directory()
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
