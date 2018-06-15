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
import random


CURRENT_DIR = os.getcwd()
WORKFLOW_FILE_PATH = CURRENT_DIR + "/data/workflow_connections_merged.tsv"
WORKFLOW_PATHS_FILE_DUP = CURRENT_DIR + "/data/workflow_connections_duplicate_paths.txt"
WORKFLOW_PATHS_FREQ = CURRENT_DIR + "/data/workflow_paths_freq.txt"
WORKFLOW_PATHS_FILE = CURRENT_DIR + "/data/workflow_connections_paths.txt"
COMPATIBLE_NEXT_TOOLS = CURRENT_DIR + "/data/compatible_tools.json"


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
        workflow_paths_dup = ""
        workflow_paths_unique = ""
        workflow_parents = dict()
        workflow_paths = list()
        unique_paths = list()
        workflow_paths_freq = dict()
        with open( WORKFLOW_FILE_PATH, 'rt' ) as workflow_connections_file:
            workflow_connections = csv.reader( workflow_connections_file, delimiter=',' )
            for index, row in enumerate( workflow_connections ):
                if not index:
                    continue  
                wf_id = str( row[ 0 ] ) 
                if wf_id not in workflows:
                    workflows[ wf_id ] = list()
                in_tool = row[ 2 ]
                out_tool = row[ 5 ]
                if out_tool and in_tool and out_tool != in_tool:
                    in_tool = self.format_tool_id( in_tool )
                    out_tool = self.format_tool_id( out_tool )
                    workflows[ wf_id ].append( ( in_tool, out_tool ) )
        print( "Processing workflows..." )
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
        print( "Workflows processed" )
        print( "All paths: %d" % len( workflow_paths ) )
        print( "Counting frequency of paths..." )
        for path in workflow_paths:
            path_names = ",".join( path )
            if path_names not in workflow_paths_freq:
                  workflow_paths_freq[ path_names ] = 0
            workflow_paths_freq[ path_names ] += 1
        with open( WORKFLOW_PATHS_FREQ , "w" ) as workflow_paths_freq_file:
            workflow_paths_freq_file.write( json.dumps( workflow_paths_freq ) )
        # collect duplicate paths
        for path in workflow_paths:
            workflow_paths_dup += ",".join( path ) + "\n"
        with open( WORKFLOW_PATHS_FILE_DUP, "w" ) as workflows_file:
            workflows_file.write( workflow_paths_dup )
        # collect unique paths
        print( "Removing duplicate paths..." )
        unique_paths = list( set( workflow_paths_dup.split( "\n" ) ) )
        print( "Unique paths: %d" % len( unique_paths ) )
        print( "Finding compatible next tools..." )
        next_tools = self.set_compatible_next_tools( unique_paths )
        with open( COMPATIBLE_NEXT_TOOLS , "w" ) as compatible_tools_file:
            compatible_tools_file.write( json.dumps( next_tools ) )
        print( "Writing workflows to a text file..." )
        random.shuffle( unique_paths )
        for path in unique_paths:
            workflow_paths_unique += path + "\n"
        with open( WORKFLOW_PATHS_FILE, "w" ) as workflows_file:
            workflows_file.write( workflow_paths_unique )
            
    @classmethod
    def set_compatible_next_tools( self, workflow_paths ):
        """
        Find next tools for each tool
        """
        next_tools = dict()
        for path in workflow_paths:
            path_split = path.split( "," )
            for window in range( 0, len( path_split ) - 1 ):
                current_next_tools = path_split[ window: window + 2 ]
                current_tool = current_next_tools[ 0 ]
                next_tool = current_next_tools[ 1 ]
                if current_tool in next_tools:
                    next_tools[ current_tool ] += "," + next_tool
                else:
                    next_tools[ current_tool ] = next_tool
        for tool in next_tools:
            next_tools[ tool ] = ",".join( list( set( next_tools[ tool ].split( "," ) ) ) )
        return next_tools

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
        children = graph.keys()
        roots = list( set( all_parents).difference( set( children ) ) )
        leaves = list( set( children ).difference( set( all_parents ) ) )
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
        tool_id = tool_id.replace( ":", "" )
        return tool_id.lower()
