"""
Extract workflow paths from the tabular file containing
input and output tools
"""

import csv
import random


class ExtractWorkflowConnections:

    @classmethod
    def __init__( self ):
        """ Init method. """

    @classmethod
    def read_tabular_file( self, raw_file_path ):
        """
        Read tabular file and extract workflow connections
        """
        print( "Reading workflows..." )
        workflows = {}
        workflow_paths_dup = ""
        workflow_parents = dict()
        workflow_paths = list()
        unique_paths = list()
        tool_name_display = dict()
        with open( raw_file_path, 'rt' ) as workflow_connections_file:
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
                    in_tool_original, in_tool = self.format_tool_id( in_tool )
                    out_tool_original, out_tool = self.format_tool_id( out_tool )
                    workflows[ wf_id ].append( ( in_tool, out_tool ) )
                    if in_tool not in tool_name_display:
                        tool_name_display[in_tool] = in_tool_original
                    if out_tool not in tool_name_display:
                        tool_name_display[out_tool] = out_tool_original

        print( "Processing workflows..." )
        wf_ctr = 0
        for wf_id in workflows:
            wf_ctr += 1
            workflow_parents[ wf_id ] = self.read_workflow( wf_id, workflows[ wf_id ] )
        for wf_id in workflow_parents:
            flow_paths = list()
            parents_graph = workflow_parents[ wf_id ]
            roots, leaves = self.get_roots_leaves( parents_graph )
            for root in roots:
                for leaf in leaves:
                    paths = self.find_tool_paths_workflow( parents_graph, root, leaf )
                    # reverse the paths as they are computed from leaves to roots leaf
                    paths = [ tool_path for tool_path in paths ]
                    if len( paths ) > 0:
                        flow_paths.extend( paths )
            workflow_paths.extend( flow_paths )

        print( "Workflows processed: %d" % wf_ctr )
        print( "# paths in workflows: %d" % len( workflow_paths ) )

        # collect duplicate paths
        for path in workflow_paths:
            workflow_paths_dup += ",".join( path ) + "\n"

        # collect unique paths
        unique_paths = list( workflow_paths_dup.split( "\n" ) )
        unique_paths = list(filter(None, unique_paths))
        print( "Unique paths: %d" % len( unique_paths ) )

        random.shuffle( unique_paths )
        
        print( "Finding compatible next tools..." )
        compatible_next_tools = self.set_compatible_next_tools(unique_paths)
        return unique_paths, compatible_next_tools

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
        return tool_id, tool_id
