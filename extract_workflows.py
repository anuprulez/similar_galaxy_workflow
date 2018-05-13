"""
Extract steps, tools and input and output types of workflows.
"""
import sys
import os
import json
import time
import pandas as pd
import operator
from itertools import groupby


class ExtractWorkflows:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.workflow_directory = 'data/workflows/'
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
                    tool_input_connections = dict()
                    tool_output_connections = dict()
                    for step in all_steps:
                        wf_step = all_steps[ step ]
                        steps.append( { "id": wf_step[ "id" ], "name": wf_step[ "name" ], "tool_id": wf_step[ "tool_id" ], "input_connections": wf_step[ "input_connections" ], "type": wf_step[ "type" ], "tool_state": json.loads( wf_step[ "tool_state" ] ), "label": wf_step[ "label" ], "outputs": wf_step[ "outputs" ] } )
                    steps = sorted( steps, key=operator.itemgetter( "id" ) )
                    for step in steps:
                        # take a workflow if there is at least one step
                        tool_id_orig = step[ "tool_id" ]
                        if tool_id_orig:
                            parent_ids = list()
                            parent_outputs = list()
                            output_types = list()
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
                                    parent_outputs.append( parents[ item ][ "output_name" ] )
                            immediate_parents[ str( tool_internal_id ) ] = list( set( parent_ids ) )
                            tool_input_connections[ str( tool_internal_id ) ] = parent_outputs
                            for output in step[ "outputs" ]:
                                output_types.append( { "name": output[ "name" ], "type": output[ "type" ] } )
                            tool_output_connections[ str( tool_internal_id ) ] = output_types
                    if len( tool_steps ) > 0:
                        workflow_json[ "steps" ] = tool_steps
                        workflow_json[ "id" ] = file_id
                        workflow_json[ "name" ] = file_json[ "name" ]
                        workflow_json[ "tags" ] = ",".join( file_json[ "tags" ] )
                        workflow_json[ "annotation" ] = file_json[ "annotation" ]
                        workflow_json[ "parents" ] = immediate_parents
                        workflow_json[ "original_steps" ] = steps
                        workflow_json[ "input_types" ] = tool_input_connections
                        workflow_json[ "output_types" ] = tool_output_connections
            except Exception as exception:
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

        workflows_to_json = dict()
        for item in workflow_json:
            workflows_to_json[ item[ "id" ] ] = item

        with open( "data/workflows.json", "w" ) as workflows_as_json:
            workflows_as_json.write( json.dumps( workflows_to_json ) )
            workflows_as_json.close()
            
        workflow_seqes = wf_steps_sentences.split( "\n" )
        wf_steps_sentences = list( set( workflow_seqes ) )
        wf_steps_sentences = "\n".join( wf_steps_sentences )

        # write all the paths from all the workflow to a text file
        with open( "data/workflow_steps.txt", "w" ) as steps_txt:
            steps_txt.write( wf_steps_sentences )
            steps_txt.close()

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
                    # merge the repeated tools into just one tool
                    last_tool_name = sequence.split( " " )[ -1 ]
                    if last_tool_name != tool_name:
                        sequence += " " + tool_name
            sequence += "\n"
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

    @classmethod
    def assign_input_output_types( self, wf_json_path ):
        """
        Assign input and output types to tools
        """
        workflows = list()
        with open( wf_json_path, "r" ) as wf_json:
            workflows = json.loads( wf_json.read() )
        tool_io_types = dict()
        for wf_id in workflows:
            wf = workflows[ wf_id ]
            for step in wf[ "steps" ]:
                tool_name = wf[ "steps" ][ step ]
                tool_input_types = list()
                if not tool_name in tool_io_types:
                    tool_io_types[ tool_name ] = dict()

                ip_types = wf[ "input_types" ][ step ]
                parents = wf[ "parents" ][ step ]
                for it in ip_types:
                    for parent_id in parents:
                       ot = wf[ "output_types" ][ parent_id ]
                       if len( ot ) > 0 and it == ot[ 0 ][ "name" ]:
                           ip_type = ot[ 0 ][ "type" ]
                           if ip_type != "input":
                               tool_input_types.append( ip_type )
                if "input_types" in tool_io_types[ tool_name ]:
                    tool_io_types[ tool_name ][ "input_types" ].extend( tool_input_types )
                else:
                   tool_io_types[ tool_name ][ "input_types" ] = tool_input_types

                # set output types. If there is "input" in the output types, copy all items from the input types
                outtps = [ obj[ "type" ] for obj in wf[ "output_types" ][ step ] ]
                if "input" in outtps:
                    outtps.extend( tool_input_types )
                outtps = [ tp for tp in outtps if tp != "input" ]
                if "output_types" in tool_io_types[ tool_name ]:
                    tool_io_types[ tool_name ][ "output_types" ].extend( outtps )
                else:
                    tool_io_types[ tool_name ][ "output_types" ] = outtps

                tool_io_types[ tool_name ][ "input_types" ] = list( set( tool_io_types[ tool_name ][ "input_types" ] ) )
                tool_io_types[ tool_name ][ "output_types" ] = list( set( tool_io_types[ tool_name ][ "output_types" ] ) )
        with open( "data/tools_file_types.json", "w" ) as tool_types:
            tool_types.write( json.dumps( tool_io_types ) ) 


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python extract_workflows.py" )
        exit( 1 )
    start_time = time.time()
    extract_workflow = ExtractWorkflows()
    extract_workflow.read_workflow_directory()
    print( "Finished extracting workflows" )
    extract_workflow.assign_input_output_types( "data/workflows.json" )
    end_time = time.time()
    print ("Program finished in %d seconds" % int( end_time - start_time ) )
