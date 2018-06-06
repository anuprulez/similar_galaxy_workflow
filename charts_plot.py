import json
import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 26
plt.rcParams["font.family"] = "FreeSerif"
plt.rc('text', usetex=True)
plt.rcParams[ 'text.latex.preamble' ]=[r"\usepackage{amsmath}"]
plt.rcParams[ "font.size" ] = FONT_SIZE


def plot_tools_compatible_tools( file_path ):
    next_tools = list()
    with open( file_path, 'r' ) as file_next_tools:
        next_tools_list = json.loads( file_next_tools.read() )
    for tool in next_tools_list:
        next_tools.append( len( next_tools_list[ tool ].split( "," ) ) )
    plt.bar( np.arange( len( next_tools ) ), next_tools )
    plt.ylabel( 'Number of next compatible tools' )
    plt.xlabel( 'Tools' )
    plt.title( 'Distribution of next compatible tools for all the tools' )
    plt.grid( True )
    plt.show()


def plot_data_distribution( file_path ):
    paths = list()
    with open( file_path, 'r' ) as dist_file:
        data = dist_file.read()
        for path in data.split( "\n" ):
            if path is not "":
                paths.append( path )
    tools_freq = list()
    count = list()
    for index, path in enumerate( paths ):
        size = len( path.split( "," ) )
        tools_freq.append( size )
        count.append( index )
    print len(tools_freq)
    plt.bar( count, tools_freq, color='r' )
    plt.xlabel( 'Workflow paths' )
    plt.ylabel( 'Number of tools (size)' )
    plt.title( 'Distribution of the size of workflow paths' )
    plt.grid( True )
    plt.show()


##plot_tools_compatible_tools( "data/compatible_tools.json" )
plot_data_distribution( "data/workflow_connections_paths.txt" )
