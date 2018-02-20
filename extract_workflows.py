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
        self.tool_data_filename = 'data/workflows_raw.csv'
        self.tools_filename = "data/all_tools.csv"
        self.workflows_filename = "data/processed_workflows.csv"

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
                    tool_steps = dict()
                    tool_internal_ids = list()
                    steps = list()
                    immediate_parents = dict()
                    for step in all_steps:
                        wf_step = all_steps[ step ]
                        steps.append( { "id": wf_step[ "id" ], "tool_id": wf_step[ "tool_id" ], "input_connections": wf_step[ "input_connections" ] } )
                    steps = sorted( steps, key=operator.itemgetter( "id" ) )
                    for step in steps:
                        # take a workflow if there is at least one step
                        tool_id_orig = step[ "tool_id" ]
                        if tool_id_orig:
                            parent_ids = list()
                            tool_id = self.extract_tool_id( tool_id_orig )
                            tool_internal_id = step[ "id" ]
                            tool_internal_ids.append( tool_internal_id )
                            tool_steps[ str( tool_internal_id ) ] = tool_id
                            workflow_tools.append( tool_id_orig )
                            parents = step[ "input_connections" ]
                            for item in parents:
                                parent_id = parents[ item ][ "id" ]
                                # take only those parents whose tool id is not null
                                if steps[ parent_id ][ "tool_id" ] is not None:
                                    parent_ids.append( str( parent_id ) )
                            immediate_parents[ str( tool_internal_id ) ] = list( set( parent_ids ) )
                    if len( tool_steps ) > 0:
                        workflow_json[ "steps" ] = tool_steps
                        workflow_json[ "id" ] = file_id
                        workflow_json[ "name" ] = file_json[ "name" ]
                        workflow_json[ "tags" ] = ",".join( file_json[ "tags" ] )
                        workflow_json[ "annotation" ] = file_json[ "annotation" ]
                        workflow_json[ "parents" ] = immediate_parents
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

        # create flow paths from all workflows and write them as sentences
        wf_steps_sentences = ""
        for item in workflow_json:
            flow_paths = list()
            parents_graph = item[ "parents" ]
            steps = item[ "steps" ]
            roots, leaves = self.get_roots_leaves( parents_graph )
            for root in roots:
                for leaf in leaves:
                    paths = self.find_tool_paths_workflow( parents_graph, str( root ), str( leaf ) )
                    # reverse the paths as they are computed from leaves to roots
                    paths = [ list( reversed( tool_path ) ) for tool_path in paths ]
                    if len( paths ) > 0:
                        flow_paths.extend( paths )
            all_tool_paths = self.tool_seq_toolnames( steps, flow_paths )
            if wf_steps_sentences == "":
                wf_steps_sentences = all_tool_paths
            else:
                wf_steps_sentences += all_tool_paths
        # write all the paths from all the workflow to a text file
        with open( "data/workflow_steps.txt", "w" ) as steps_txt:
            steps_txt.write( wf_steps_sentences )

    @classmethod
    def process_tool_names( self, tool_name ):
        if " " in tool_name:
            tool_name = tool_name.replace( " ", "_" )
        return tool_name.lower()

    @classmethod
    def tool_seq_toolnames( self, tool_dict, paths ):
        tool_seq = ""
        for path in paths:
            # create tool paths
            sequence = ""
            for tool in path:
                tool_name = self.process_tool_names( tool_dict[ tool ] )
                if sequence == "":
                    sequence = tool_name
                else:
                    sequence += " " + tool_name
            sequence += " . "
            # exclude the duplicate tool paths
            if sequence not in tool_seq:
                if tool_seq == "":
                    tool_seq = sequence
                else:
                    tool_seq += sequence
        return tool_seq

    @classmethod
    def find_tool_paths_workflow( self, graph, start, end, path=[] ):
        path = path + [ end ]
        if start == end:
            return [ path ]
        path_list = list()
        for node in graph[ end ]:
            if node not in path:
                new_tools_paths = self.find_tool_paths_workflow( graph, start, node, path )
                for tool_path in new_tools_paths:
                    path_list.append( tool_path )
        return path_list

    @classmethod
    def get_roots_leaves( self, graph ):
        roots = list()
        leaves = list()
        all_parents = list()
        for item in graph:
            all_parents.extend( graph[ item ] )
        all_parents = list( set( all_parents ) )

        for item in graph:
            if len( graph[ item ] ) == 0 and item in all_parents:
                roots.append( item )
            if  len( graph[ item ] ) > 0 and item not in all_parents:
                leaves.append( item )
        return roots, leaves

    @classmethod
    def extract_tool_id( self, tool_link ):
        tool_id_split = tool_link.split( "/" )
        tool_id = tool_id_split[ -2 ] if len( tool_id_split ) > 1 else tool_link
        tool_id_split = tool_id.split( "." )
        return tool_id_split[ 0 ] if len( tool_id ) > 1 else tool_id


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python extract_workflows.py" )
        exit( 1 )
    start_time = time.time()
    extract_workflow = ExtractWorkflows()
    extract_workflow.read_workflow_directory()
    end_time = time.time()
    print "Program finished in %d seconds" % int( end_time - start_time )
