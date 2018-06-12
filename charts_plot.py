import json
import matplotlib.pyplot as plt
import numpy as np
import json


FONT_SIZE = 26
plt.rcParams["font.family"] = "FreeSerif"
plt.rc('text', usetex=True)
plt.rcParams[ 'text.latex.preamble' ]=[r"\usepackage{amsmath}"]
plt.rcParams[ "font.size" ] = FONT_SIZE


def read_file( file_path ):
    content = None
    with open( file_path, 'r' ) as data_file:
        content = data_file.read()
        content = content.split( "\n" )
    content = [ float( item ) for item in content if item is not '' ]
    return content


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


def plot_activation_perf(  ):
    NEW_FONT_SIZE = FONT_SIZE - 6
    mean_test_abs_precision_relu = read_file( "thesis_results/activations/relu/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_relu = read_file( "thesis_results/activations/relu/mean_test_compatibility_precision.txt" )
    mean_train_loss_relu = read_file( "thesis_results/activations/relu/mean_train_loss.txt" )
    mean_val_loss_relu = read_file( "thesis_results/activations/relu/mean_test_loss.txt" )

    mean_test_abs_precision_tanh = read_file( "thesis_results/activations/tanh/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_tanh = read_file( "thesis_results/activations/tanh/mean_test_compatibility_precision.txt" )
    mean_train_loss_tanh = read_file( "thesis_results/activations/tanh/mean_train_loss.txt" )
    mean_val_loss_tanh = read_file( "thesis_results/activations/tanh/mean_test_loss.txt" )

    mean_test_abs_precision_sigmoid = read_file( "thesis_results/activations/sigmoid/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_sigmoid = read_file( "thesis_results/activations/sigmoid/mean_test_compatibility_precision.txt" )
    mean_train_loss_sigmoid = read_file( "thesis_results/activations/sigmoid/mean_train_loss.txt" )
    mean_val_loss_sigmoid = read_file( "thesis_results/activations/sigmoid/mean_test_loss.txt" )

    mean_test_abs_precision_elu = read_file( "data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_elu = read_file( "data/mean_test_compatibility_precision.txt" )
    mean_train_loss_elu = read_file( "data/mean_train_loss.txt" )
    mean_val_loss_precision_elu = read_file( "data/mean_test_loss.txt" )
    title = "Precision and loss for various activations"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "relu", 'tanh', 'sigmoid', 'elu' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        print row, axis
        if row == 0:
            axis[ 0 ].plot( mean_test_abs_precision_relu )
            axis[ 0 ].plot( mean_test_abs_precision_tanh )
            axis[ 0 ].plot( mean_test_comp_precision_sigmoid )
            axis[ 0 ].plot( mean_test_abs_precision_relu )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            
            axis[ 1 ].plot( mean_test_comp_precision_relu )
            axis[ 1 ].plot( mean_test_comp_precision_tanh )
            axis[ 1 ].plot( mean_test_comp_precision_sigmoid )
            axis[ 1 ].plot( mean_test_comp_precision_elu )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            axis[ 0 ].plot( mean_train_loss_relu )
            axis[ 0 ].plot( mean_train_loss_tanh )
            axis[ 0 ].plot( mean_train_loss_sigmoid )
            axis[ 0 ].plot( mean_train_loss_elu )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )

            axis[ 1 ].plot( mean_val_loss_relu )
            axis[ 1 ].plot( mean_val_loss_tanh )
            axis[ 1 ].plot( mean_val_loss_sigmoid )
            axis[ 1 ].plot( mean_val_loss_precision_elu )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
    plt.suptitle( title )
    plt.show()

plot_activation_perf()
##plot_tools_compatible_tools( "data/compatible_tools.json" )
##plot_data_distribution( "data/workflow_connections_paths.txt" )
