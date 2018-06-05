"""
Extract steps, tools and input and output types of workflows.
"""
import sys
import os
import json
import time
import pandas as pd
import operator
import csv


CURRENT_DIR = os.getcwd()
WORKFLOW_FILE_PATH = CURRENT_DIR + "/data/workflows_connections/workflow_connections.tsv"


class ExtractWorkflowConnections:

    @classmethod
    def __init__( self ):
        """ Init method. """

    @classmethod
    def read_tabular_file( self ):
        """
        Read tabular file and extract workflow connections
        """
        print( "Reading workflows..." )
        workflows = {}
        with open( WORKFLOW_FILE_PATH, 'rb' ) as workflow_connections_file:
            workflow_connections = csv.reader( workflow_connections_file, delimiter=',' )
            for index, row in enumerate( workflow_connections ):
                if not index:
                    continue  
                row_split = row[ 0 ].split( "\t" )
                wf_id = str( row_split[ 0 ] ) 
                if wf_id not in workflows:
                    workflows[ wf_id ] = list()
                in_tool = row_split[ 2 ]
                out_tool = row_split[ 5 ]
                if out_tool and in_tool and out_tool != in_tool:
                    in_tool = self.format_tool_id( in_tool )
                    out_tool = self.format_tool_id( out_tool )
                    if ( in_tool, out_tool ) not in workflows:
                        workflows[ wf_id ].append( ( in_tool, out_tool ) )

        print( "Processing workflows..." )
        workflow_parents = dict()
        workflow_paths = list()
        for wf_id in workflows:
            workflow_parents[ wf_id ] = self.read_workflow( wf_id, workflows[ wf_id ] )
        for wf_id in workflow_parents:
            flow_paths = list()
            parents_graph = workflow_parents[ wf_id ]
            roots, leaves = self.get_roots_leaves( parents_graph )
            for root in roots:
                for leaf in leaves:
                    paths = self.find_tool_paths_workflow( parents_graph, root, leaf )
                    # reverse the paths as they are computed from leaves to roots
                    paths = [ list( reversed( tool_path ) ) for tool_path in paths ]
                    if len( paths ) > 0:
                        flow_paths.extend( paths )
            workflow_paths.extend( flow_paths )
        unique_paths = list()
        print( "Paths processed" )
        for path in workflow_paths:
            if path not in unique_paths:
                unique_paths.append( path )
        print len( unique_paths )

    @classmethod
    def read_workflow( self, wf_id, workflow_rows ):
        """
        Read all connections for a workflow
        """
        tool_parents = dict()
        for connection in workflow_rows:
            in_tool = connection[ 0 ]
            out_tool = connection[ 1 ]
            if out_tool not in tool_parents:
                tool_parents[ out_tool ] = list()
            if in_tool not in tool_parents[ out_tool ]:
                tool_parents[ out_tool ].append( in_tool )
        return tool_parents

    @classmethod
    def get_roots_leaves( self, graph ):
        roots = list()
        leaves = list()
        all_parents = list()
        for item in graph:
            all_parents.extend( graph[ item ] )
        all_parents = list( set( all_parents ) )
        keys = graph.keys()
        for parent in all_parents:
            if parent not in keys:
                roots.append( parent )
        for key in keys:
            if key not in all_parents:
                leaves.append( key )       
        return roots, leaves 

    @classmethod
    def find_tool_paths_workflow( self, graph, start, end, path=[] ):
        path = path + [ end ]
        if start == end:
            return [ path ]
        path_list = list()
        if end in graph:
            for node in graph[ end ]:
                if node not in path:
                    new_tools_paths = self.find_tool_paths_workflow( graph, start, node, path )
                    for tool_path in new_tools_paths:
                        path_list.append( tool_path )
        return path_list

    @classmethod
    def format_tool_id( self, tool_link ):
        tool_id_split = tool_link.split( "/" )
        tool_id = tool_id_split[ -2 ] if len( tool_id_split ) > 1 else tool_link
        tool_id_split = tool_id.split( "." )
        tool_id = tool_id_split[ 0 ] if len( tool_id ) > 1 else tool_id
        tool_id = tool_id.replace( " ", "_" )
        return tool_id.lower()


if __name__ == "__main__":
    if len(sys.argv) != 1:
        print( "Usage: python extract_workflow_connections.py" )
        exit( 1 )
    start_time = time.time()
    connections = ExtractWorkflowConnections()
    connections.read_tabular_file()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
