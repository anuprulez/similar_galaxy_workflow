"""
Extract steps, tools and input and output types of workflows.
"""
import sys
import os
import json
import time
import pandas as pd
import operator


class ExtractWorkflows:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.workflow_directory = 'data/workflows/'
        self.tool_data_filename = 'workflows_raw.csv'
        self.tools_filename = "all_tools.csv"
        self.workflows_filename = "processed_workflows.csv"

    @classmethod
    def read_workflow_file( self, workflow_file_path, file_id ):
        """
        Read a workflow file to collect all the steps and other attributes
        """
        workflow_tools = list()
        workflow_json = dict()
        with open( workflow_file_path, 'r' ) as workflow_file:
            file_raw = workflow_file.readline()
            try:
                file_json = json.loads( file_raw )
                if "err_msg" not in file_json:
                    all_steps = file_json[ "steps" ]
                    tool_steps = list()
                    steps = list()
                    for step in all_steps:
                        wf_step = all_steps[ step ]
                        steps.append( { "id": wf_step[ "id" ], "tool_id": wf_step[ "tool_id" ] } )
                    steps = sorted( steps, key=operator.itemgetter( "id" ) )
                    for step in steps:
                        # take a workflow if there is at least one step
                        tool_id_orig = step[ "tool_id" ]
                        if tool_id_orig:
                            tool_id = self.extract_tool_id( tool_id_orig )
                            tool_steps.append( tool_id )
                            workflow_tools.append( tool_id_orig )
                    if len( tool_steps ) > 0:
                        workflow_json[ "steps" ] = ",".join( tool_steps )
                        workflow_json[ "id" ] = file_id
                        workflow_json[ "name" ] = file_json[ "name" ]
                        workflow_json[ "tags" ] = ",".join( file_json[ "tags" ] )
                        workflow_json[ "annotation" ] = file_json[ "annotation" ]
            except Exception:
                pass
        return workflow_json, workflow_tools

    @classmethod
    def read_workflow_directory( self ):
        """
        Read workflow's directory
        """
        workflow_json = list()
        all_workflow_tools = list()
        all_workflow_tools_id = list()
        tools = list()
        for folder in os.listdir( self.workflow_directory ):
            workflow_path = os.path.join( self.workflow_directory, folder )
            for workflow_file_id in os.listdir( workflow_path ):
                wf, tools = self.read_workflow_file( os.path.join( workflow_path, workflow_file_id ), workflow_file_id )
                if workflow_file_id not in workflow_json and wf:
                    workflow_json.append( wf )
                all_workflow_tools.extend( tools )
        all_workflow_tools = list( set( all_workflow_tools ) )
        # extract ids from the tool ids link
        for item in all_workflow_tools:
            tool_id = self.extract_tool_id( item )
            if tool_id not in tools:
                all_workflow_tools_id.append( { "Original id": item, "Tool id": tool_id } )
                tools.append( tool_id )
        # write all the unique tools to a tabular file
        all_tools_dataframe = pd.DataFrame( all_workflow_tools_id )
        all_tools_dataframe.to_csv( self.tools_filename, encoding='utf-8' )
        # write all the workflows to a tabular file
        all_workflows_dataframe = pd.DataFrame( workflow_json )
        all_workflows_dataframe.to_csv( self.workflows_filename, encoding='utf-8' )

    @classmethod
    def extract_tool_id( self, tool_link ):
        tool_id_split = tool_link.split( "/" )
        return tool_id_split[ -2 ] if len( tool_id_split ) > 1 else tool_link


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python extract_workflows.py" )
        exit( 1 )
    start_time = time.time()
    extract_workflow = ExtractWorkflows()
    extract_workflow.read_workflow_directory()
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
