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

def read_json( file_path ):
    file_content = None
    with open( file_path, 'r' ) as file_json:
        file_content = json.loads( file_json.read() )
    return file_content

def plot_tools_compatible_tools( file_path ):
    next_tools = list()
    with open( file_path, 'r' ) as file_next_tools:
        next_tools_list = json.loads( file_next_tools.read() )
    for tool in next_tools_list:
        next_tools.append( len( next_tools_list[ tool ].split( "," ) ) )
    plt.bar( np.arange( len( next_tools ) ), next_tools, color='r' )
    plt.ylabel( 'Number of tools' )
    plt.xlabel( 'Tools' )
    plt.title( 'Distribution of number of next compatible tools' )
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
    plt.bar( count, tools_freq, color='r' )
    plt.xlabel( 'Count of workflow paths' )
    plt.ylabel( 'Number of tools' )
    plt.title( 'Distribution of number of tools in workflow paths' )
    plt.grid( True )
    plt.show()


def plot_activation_perf(  ):
    NEW_FONT_SIZE = FONT_SIZE - 2
    mean_test_abs_precision_relu = read_file( "thesis_results_reverse/activations/relu/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_relu = read_file( "thesis_results_reverse/activations/relu/mean_test_compatibility_precision.txt" )
    mean_train_loss_relu = read_file( "thesis_results_reverse/activations/relu/mean_train_loss.txt" )
    mean_val_loss_relu = read_file( "thesis_results_reverse/activations/relu/mean_test_loss.txt" )

    mean_test_abs_precision_tanh = read_file( "thesis_results_reverse/activations/tanh/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_tanh = read_file( "thesis_results_reverse/activations/tanh/mean_test_compatibility_precision.txt" )
    mean_train_loss_tanh = read_file( "thesis_results_reverse/activations/tanh/mean_train_loss.txt" )
    mean_val_loss_tanh = read_file( "thesis_results_reverse/activations/tanh/mean_test_loss.txt" )

    mean_test_abs_precision_sigmoid = read_file( "thesis_results_reverse/activations/sigmoid/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_sigmoid = read_file( "thesis_results_reverse/activations/sigmoid/mean_test_compatibility_precision.txt" )
    mean_train_loss_sigmoid = read_file( "thesis_results_reverse/activations/sigmoid/mean_train_loss.txt" )
    mean_val_loss_sigmoid = read_file( "thesis_results_reverse/activations/sigmoid/mean_test_loss.txt" )

    mean_test_abs_precision_elu = read_file( "thesis_results_reverse/activations/elu/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_elu = read_file( "thesis_results_reverse/activations/elu/mean_test_compatibility_precision.txt" )
    mean_train_loss_elu = read_file( "thesis_results_reverse/activations/elu/mean_train_loss.txt" )
    mean_val_loss_precision_elu = read_file( "thesis_results_reverse/activations/elu/mean_test_loss.txt" )
    title = "Precision and loss for various activations"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "relu", 'tanh', 'sigmoid', 'elu' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_relu )
            axis[ 0 ].plot( mean_test_abs_precision_tanh )
            axis[ 0 ].plot( mean_test_abs_precision_sigmoid )
            axis[ 0 ].plot( mean_test_abs_precision_elu )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_relu )
            axis[ 1 ].plot( mean_test_comp_precision_tanh )
            axis[ 1 ].plot( mean_test_comp_precision_sigmoid )
            axis[ 1 ].plot( mean_test_comp_precision_elu )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_relu )
            axis[ 0 ].plot( mean_train_loss_tanh )
            axis[ 0 ].plot( mean_train_loss_sigmoid )
            axis[ 0 ].plot( mean_train_loss_elu )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_relu )
            axis[ 1 ].plot( mean_val_loss_tanh )
            axis[ 1 ].plot( mean_val_loss_sigmoid )
            axis[ 1 ].plot( mean_val_loss_precision_elu )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_optimiser_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2
    mean_test_abs_precision_sgd = read_file( "thesis_results_reverse/optimiser/sgd/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_sgd = read_file( "thesis_results_reverse/optimiser/sgd/mean_test_compatibility_precision.txt" )
    mean_train_loss_sgd = read_file( "thesis_results_reverse/optimiser/sgd/mean_train_loss.txt" )
    mean_val_loss_sgd = read_file( "thesis_results_reverse/optimiser/sgd/mean_test_loss.txt" )

    mean_test_abs_precision_adagrad = read_file( "thesis_results_reverse/optimiser/adagrad/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_adagrad = read_file( "thesis_results_reverse/optimiser/adagrad/mean_test_compatibility_precision.txt" )
    mean_train_loss_adagrad = read_file( "thesis_results_reverse/optimiser/adagrad/mean_train_loss.txt" )
    mean_val_loss_adagrad = read_file( "thesis_results_reverse/optimiser/adagrad/mean_test_loss.txt" )

    mean_test_abs_precision_adam = read_file( "thesis_results_reverse/optimiser/adam/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_adam = read_file( "thesis_results_reverse/optimiser/adam/mean_test_compatibility_precision.txt" )
    mean_train_loss_adam = read_file( "thesis_results_reverse/optimiser/adam/mean_train_loss.txt" )
    mean_val_loss_adam = read_file( "thesis_results_reverse/optimiser/adam/mean_test_loss.txt" )

    mean_test_abs_precision_rmsprop = read_file( "thesis_results_reverse/optimiser/rmsprop/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_rmsprop = read_file( "thesis_results_reverse/optimiser/rmsprop/mean_test_compatibility_precision.txt" )
    mean_train_loss_rmsprop = read_file( "thesis_results_reverse/optimiser/rmsprop/mean_train_loss.txt" )
    mean_val_loss_precision_rmsprop = read_file( "thesis_results_reverse/optimiser/rmsprop/mean_test_loss.txt" )
    title = "Precision and loss for various optimisers"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "SGD", 'Adagrad', 'Adam', 'RMSProp' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_sgd )
            axis[ 0 ].plot( mean_test_abs_precision_adagrad )
            axis[ 0 ].plot( mean_test_abs_precision_adam )
            axis[ 0 ].plot( mean_test_abs_precision_rmsprop )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_sgd )
            axis[ 1 ].plot( mean_test_comp_precision_adagrad )
            axis[ 1 ].plot( mean_test_comp_precision_adam )
            axis[ 1 ].plot( mean_test_comp_precision_rmsprop )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_sgd )
            axis[ 0 ].plot( mean_train_loss_adagrad )
            axis[ 0 ].plot( mean_train_loss_adam )
            axis[ 0 ].plot( mean_train_loss_rmsprop )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_sgd )
            axis[ 1 ].plot( mean_val_loss_adagrad )
            axis[ 1 ].plot( mean_val_loss_adam )
            axis[ 1 ].plot( mean_val_loss_precision_rmsprop )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 1 ].grid( True )
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


def plot_lr_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2
    mean_test_abs_precision_01 = read_file( "thesis_results_reverse/lr/0.01/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_01 = read_file( "thesis_results_reverse/lr/0.01/mean_test_compatibility_precision.txt" )
    mean_train_loss_01 = read_file( "thesis_results_reverse/lr/0.01/mean_train_loss.txt" )
    mean_val_loss_01 = read_file( "thesis_results_reverse/lr/0.01/mean_test_loss.txt" )

    mean_test_abs_precision_005 = read_file( "thesis_results_reverse/lr/0.005/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_005 = read_file( "thesis_results_reverse/lr/0.005/mean_test_compatibility_precision.txt" )
    mean_train_loss_005 = read_file( "thesis_results_reverse/lr/0.005/mean_train_loss.txt" )
    mean_val_loss_005 = read_file( "thesis_results_reverse/lr/0.005/mean_test_loss.txt" )

    mean_test_abs_precision_001 = read_file( "thesis_results_reverse/lr/0.001/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_001 = read_file( "thesis_results_reverse/lr/0.001/mean_test_compatibility_precision.txt" )
    mean_train_loss_001 = read_file( "thesis_results_reverse/lr/0.001/mean_train_loss.txt" )
    mean_val_loss_001 = read_file( "thesis_results_reverse/lr/0.001/mean_test_loss.txt" )

    mean_test_abs_precision_0001 = read_file( "thesis_results_reverse/lr/0.0001/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_0001 = read_file( "thesis_results_reverse/lr/0.0001/mean_test_compatibility_precision.txt" )
    mean_train_loss_0001 = read_file( "thesis_results_reverse/lr/0.0001/mean_train_loss.txt" )
    mean_val_loss_precision_0001 = read_file( "thesis_results_reverse/lr/0.0001/mean_test_loss.txt" )
    title = "Precision and loss for various learning rates"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "0.01", '0.005', '0.001', '0.0001' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_01 )
            axis[ 0 ].plot( mean_test_abs_precision_005 )
            axis[ 0 ].plot( mean_test_abs_precision_001 )
            axis[ 0 ].plot( mean_test_abs_precision_0001 )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_01 )
            axis[ 1 ].plot( mean_test_comp_precision_005 )
            axis[ 1 ].plot( mean_test_comp_precision_001 )
            axis[ 1 ].plot( mean_test_comp_precision_0001 )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_01 )
            axis[ 0 ].plot( mean_train_loss_005 )
            axis[ 0 ].plot( mean_train_loss_001 )
            axis[ 0 ].plot( mean_train_loss_0001 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_01 )
            axis[ 1 ].plot( mean_val_loss_005 )
            axis[ 1 ].plot( mean_val_loss_001 )
            axis[ 1 ].plot( mean_val_loss_precision_0001 )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 1 ].grid( True )
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


def plot_batchsize_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2
    mean_test_abs_precision_64 = read_file( "thesis_results_reverse/batchsize/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results_reverse/batchsize/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results_reverse/batchsize/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results_reverse/batchsize/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results_reverse/batchsize/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results_reverse/batchsize/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results_reverse/batchsize/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results_reverse/batchsize/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results_reverse/batchsize/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results_reverse/batchsize/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results_reverse/batchsize/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results_reverse/batchsize/256/mean_test_loss.txt" )

    mean_test_abs_precision_512 = read_file( "thesis_results_reverse/batchsize/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results_reverse/batchsize/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results_reverse/batchsize/512/mean_train_loss.txt" )
    mean_val_loss_precision_512 = read_file( "thesis_results_reverse/batchsize/512/mean_test_loss.txt" )
    title = "Precision and loss for various batch sizes"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "64", '128', '256', '512' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_64 )
            axis[ 0 ].plot( mean_test_abs_precision_128 )
            axis[ 0 ].plot( mean_test_abs_precision_256 )
            axis[ 0 ].plot( mean_test_abs_precision_512 )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_64 )
            axis[ 1 ].plot( mean_test_comp_precision_128 )
            axis[ 1 ].plot( mean_test_comp_precision_256 )
            axis[ 1 ].plot( mean_test_comp_precision_512 )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_64 )
            axis[ 0 ].plot( mean_train_loss_128 )
            axis[ 0 ].plot( mean_train_loss_256 )
            axis[ 0 ].plot( mean_train_loss_512 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_64 )
            axis[ 1 ].plot( mean_val_loss_128 )
            axis[ 1 ].plot( mean_val_loss_256 )
            axis[ 1 ].plot( mean_val_loss_precision_512 )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_dropout_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2
    mean_test_abs_precision_00 = read_file( "thesis_results_reverse/dropout/0.0/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_00 = read_file( "thesis_results_reverse/dropout/0.0/mean_test_compatibility_precision.txt" )
    mean_train_loss_00 = read_file( "thesis_results_reverse/dropout/0.0/mean_train_loss.txt" )
    mean_val_loss_00 = read_file( "thesis_results_reverse/dropout/0.0/mean_test_loss.txt" )

    mean_test_abs_precision_01 = read_file( "thesis_results_reverse/dropout/0.1/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_01 = read_file( "thesis_results_reverse/dropout/0.1/mean_test_compatibility_precision.txt" )
    mean_train_loss_01 = read_file( "thesis_results_reverse/dropout/0.1/mean_train_loss.txt" )
    mean_val_loss_01 = read_file( "thesis_results_reverse/dropout/0.1/mean_test_loss.txt" )

    mean_test_abs_precision_02 = read_file( "thesis_results_reverse/dropout/0.2/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_02 = read_file( "thesis_results_reverse/dropout/0.2/mean_test_compatibility_precision.txt" )
    mean_train_loss_02 = read_file( "thesis_results_reverse/dropout/0.2/mean_train_loss.txt" )
    mean_val_loss_02 = read_file( "thesis_results_reverse/dropout/0.2/mean_test_loss.txt" )

    mean_test_abs_precision_03 = read_file( "thesis_results_reverse/dropout/0.3/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_03 = read_file( "thesis_results_reverse/dropout/0.3/mean_test_compatibility_precision.txt" )
    mean_train_loss_03 = read_file( "thesis_results_reverse/dropout/0.3/mean_train_loss.txt" )
    mean_val_loss_03 = read_file( "thesis_results_reverse/dropout/0.3/mean_test_loss.txt" )

    mean_test_abs_precision_04 = read_file( "thesis_results_reverse/dropout/0.4/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_04 = read_file( "thesis_results_reverse/dropout/0.4/mean_test_compatibility_precision.txt" )
    mean_train_loss_04 = read_file( "thesis_results_reverse/dropout/0.4/mean_train_loss.txt" )
    mean_val_loss_04 = read_file( "thesis_results_reverse/dropout/0.4/mean_test_loss.txt" )

    title = "Precision and loss for various dropout values"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "0.0", '0.1', '0.2', '0.3', '0.4' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_00 )
            axis[ 0 ].plot( mean_test_abs_precision_01 )
            axis[ 0 ].plot( mean_test_abs_precision_02 )
            axis[ 0 ].plot( mean_test_abs_precision_03 )
            axis[ 0 ].plot( mean_test_abs_precision_04 )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_00 )
            axis[ 1 ].plot( mean_test_comp_precision_01 )
            axis[ 1 ].plot( mean_test_comp_precision_02 )
            axis[ 1 ].plot( mean_test_comp_precision_03 )
            axis[ 1 ].plot( mean_test_comp_precision_04 )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=4 )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_00 )
            axis[ 0 ].plot( mean_train_loss_01 )
            axis[ 0 ].plot( mean_train_loss_02 )
            axis[ 0 ].plot( mean_train_loss_03 )
            axis[ 0 ].plot( mean_train_loss_04 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_00 )
            axis[ 1 ].plot( mean_val_loss_01 )
            axis[ 1 ].plot( mean_val_loss_02 )
            axis[ 1 ].plot( mean_val_loss_03 )
            axis[ 1 ].plot( mean_val_loss_04 )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE, loc=1 )
            axis[ 1 ].grid( True )
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


def plot_embedding_sizes_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision_32 = read_file( "thesis_results_reverse/embeddinglayersize/32/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_32 = read_file( "thesis_results_reverse/embeddinglayersize/32/mean_test_compatibility_precision.txt" )
    mean_train_loss_32 = read_file( "thesis_results_reverse/embeddinglayersize/32/mean_train_loss.txt" )
    mean_val_loss_32 = read_file( "thesis_results_reverse/embeddinglayersize/32/mean_test_loss.txt" )

    mean_test_abs_precision_64 = read_file( "thesis_results_reverse/embeddinglayersize/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results_reverse/embeddinglayersize/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results_reverse/embeddinglayersize/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results_reverse/embeddinglayersize/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results_reverse/embeddinglayersize/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results_reverse/embeddinglayersize/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results_reverse/embeddinglayersize/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results_reverse/embeddinglayersize/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results_reverse/embeddinglayersize/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results_reverse/embeddinglayersize/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results_reverse/embeddinglayersize/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results_reverse/embeddinglayersize/256/mean_test_loss.txt" )

    mean_test_abs_precision_512 = read_file( "thesis_results_reverse/embeddinglayersize/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results_reverse/embeddinglayersize/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results_reverse/embeddinglayersize/512/mean_train_loss.txt" )
    mean_val_loss_512 = read_file( "thesis_results_reverse/embeddinglayersize/512/mean_test_loss.txt" )

    mean_test_abs_precision_1024 = read_file( "thesis_results_reverse/embeddinglayersize/1024/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_1024 = read_file( "thesis_results_reverse/embeddinglayersize/1024/mean_test_compatibility_precision.txt" )
    mean_train_loss_1024 = read_file( "thesis_results_reverse/embeddinglayersize/1024/mean_train_loss.txt" )
    mean_val_loss_1024 = read_file( "thesis_results_reverse/embeddinglayersize/1024/mean_test_loss.txt" )

    title = "Precision and loss for various sizes of embedding layer"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ '64', '128', '256', '512', '1024' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            #axis[ 0 ].plot( mean_test_abs_precision_32 )
            axis[ 0 ].plot( mean_test_abs_precision_64 )
            axis[ 0 ].plot( mean_test_abs_precision_128 )
            axis[ 0 ].plot( mean_test_abs_precision_256 )
            axis[ 0 ].plot( mean_test_abs_precision_512 )
            axis[ 0 ].plot( mean_test_abs_precision_1024 )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            #axis[ 1 ].plot( mean_test_comp_precision_32 )
            axis[ 1 ].plot( mean_test_comp_precision_64 )
            axis[ 1 ].plot( mean_test_comp_precision_128 )
            axis[ 1 ].plot( mean_test_comp_precision_256 )
            axis[ 1 ].plot( mean_test_comp_precision_512 )
            axis[ 1 ].plot( mean_test_comp_precision_1024 )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            #axis[ 0 ].plot( mean_train_loss_32 )
            axis[ 0 ].plot( mean_train_loss_64 )
            axis[ 0 ].plot( mean_train_loss_128 )
            axis[ 0 ].plot( mean_train_loss_256 )
            axis[ 0 ].plot( mean_train_loss_512 )
            axis[ 0 ].plot( mean_train_loss_1024 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            #axis[ 1 ].plot( mean_val_loss_32 )
            axis[ 1 ].plot( mean_val_loss_64 )
            axis[ 1 ].plot( mean_val_loss_128 )
            axis[ 1 ].plot( mean_val_loss_256 )
            axis[ 1 ].plot( mean_val_loss_512 )
            axis[ 1 ].plot( mean_val_loss_1024 )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_num_units_perf():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision_64 = read_file( "thesis_results_reverse/#units/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results_reverse/#units/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results_reverse/#units/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results_reverse/#units/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results_reverse/#units/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results_reverse/#units/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results_reverse/#units/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results_reverse/#units/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results_reverse/#units/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results_reverse/#units/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results_reverse/#units/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results_reverse/#units/256/mean_test_loss.txt" )
    
    mean_test_abs_precision_512 = read_file( "thesis_results_reverse/#units/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results_reverse/#units/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results_reverse/#units/512/mean_train_loss.txt" )
    mean_val_loss_512 = read_file( "thesis_results_reverse/#units/512/mean_test_loss.txt" )

    title = "Precision and loss for various number of memory units"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    legend = [ "64", '128', '256', '512' ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision_64 )
            axis[ 0 ].plot( mean_test_abs_precision_128 )
            axis[ 0 ].plot( mean_test_abs_precision_256 )
            axis[ 0 ].plot( mean_test_abs_precision_512 )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_64 )
            axis[ 1 ].plot( mean_test_comp_precision_128 )
            axis[ 1 ].plot( mean_test_comp_precision_256 )
            axis[ 1 ].plot( mean_test_comp_precision_512 )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss_64 )
            axis[ 0 ].plot( mean_train_loss_128 )
            axis[ 0 ].plot( mean_train_loss_256 )
            axis[ 0 ].plot( mean_train_loss_512 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss_64 )
            axis[ 1 ].plot( mean_val_loss_128 )
            axis[ 1 ].plot( mean_val_loss_256 )
            axis[ 1 ].plot( mean_val_loss_512 )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_extreme_paths():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision = read_file( "thesis_results_reverse/extreme_paths/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results_reverse/extreme_paths/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results_reverse/extreme_paths/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results_reverse/extreme_paths/data/mean_test_loss.txt" )

    title = "Precision and loss for decomposition of train and test paths"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision, color='r' )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision, color='r' )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss, color='r' )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss, color='r' )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_longer_paths():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision = read_file( "thesis_results_reverse/longer_train_paths/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results_reverse/longer_train_paths/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results_reverse/longer_train_paths/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results_reverse/longer_train_paths/data/mean_test_loss.txt" )

    title = "Precision and loss for no decomposition of paths"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision, color='r' )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision, color='r' )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss, color='r' )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss, color='r' )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_train_long_test_decomposed():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision = read_file( "thesis_results_reverse/train_long_test_decomposed/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results_reverse/train_long_test_decomposed/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results_reverse/train_long_test_decomposed/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results_reverse/train_long_test_decomposed/data/mean_test_loss.txt" )

    title = "Precision and loss for decomposition of only test paths"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision, color='r' )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision, color='r' )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss, color='r' )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss, color='r' )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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

def plot_top1_top2_accuracy():
    barWidth = 1
    xticks = [ "Absolute top1", "Compatible top1", "Absolute top2", "Compatible top2" ]
    xticks_repeated = [ "Absolute top-1","Absolute top-1", "Compatible top-1", "Compatible top-1", "Absolute top-2", "Absolute top-2", "Compatible top-2", "Compatible top-2" ]
    xpos = np.arange( 2 * len( xticks ) )
    data_top1_top2_test = [ 90.82140487, 98.86482131, 74.60304294, 93.94729358 ]
    data_top1_top2_train = [ 92.87385877, 99.34532873, 64.57, 90.89 ]
    xpos_1 = [ 1,3,5,7 ]
    xpos_2 = [ 2,4,6,8 ]
    plt.bar( xpos_1, data_top1_top2_test, width = barWidth, label='Test set' )
    plt.bar( xpos_2, data_top1_top2_train, width = barWidth, label='Train set' )
    plt.ylabel( 'Accuracy (in percentage)' )
    plt.xticks( [ item + 1 for item in range( len( xticks_repeated ) ) ], xticks_repeated, rotation=30 )
    plt.title( 'Absolute and compatible top-1 and top-2 accuracies' )
    plt.legend( loc=4 )
    plt.grid( True )
    plt.show()


def plot_input_length_precision():
    test_abs_topk = read_json( "data/test_input_seq_topk.json" )
    test_comp_topk = read_json( "data/test_input_seq_topk_compatible.json" )
    train_abs_topk = read_json( "data/train_input_seq_topk.json" )
    train_comp_topk = read_json( "data/train_input_seq_topk_compatible.json" )
    test_abs_topk_average = np.zeros( [ len( test_abs_topk ) ] )
    test_comp_topk_average = np.zeros( [ len( test_abs_topk ) ] )
    train_abs_topk_average = np.zeros( [ len( train_abs_topk ) ] )
    train_comp_topk_average = np.zeros( [ len( train_abs_topk ) ] )
    x_pos_test = [ item + 1 for item in range( len( test_abs_topk_average ) ) ]
    for item in test_abs_topk:
        test_abs_topk_average[ int( item ) - 1 ] = np.mean( test_abs_topk[ item ] )
    for item in test_comp_topk:
        test_comp_topk_average[ int( item ) - 1 ] = np.mean( test_comp_topk[ item ] )
    for item in train_abs_topk:
        train_abs_topk_average[ int( item ) - 1 ] = np.mean( train_abs_topk[ item ] )
    for item in train_comp_topk:
        train_comp_topk_average[ int( item ) - 1 ] = np.mean( train_comp_topk[ item ] )
    plt.plot( x_pos_test, test_abs_topk_average )
    plt.plot( x_pos_test, test_comp_topk_average )
    plt.plot( x_pos_test, train_abs_topk_average )
    plt.plot( x_pos_test, train_comp_topk_average )
    plt.xlabel( 'Length of paths' )
    plt.ylabel( 'Precision' )
    plt.title( 'Variation of precision with the length of paths' )
    plt.legend( [ "Absolute precision (test paths)", "Compatible precision (test paths)", "Absolute precision (train paths)", "Compatible precision (train paths)" ], loc=4 )
    plt.grid( True )
    plt.show()


def plot_other_classifier():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision = read_file( "thesis_results_reverse/other_classifier/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results_reverse/other_classifier/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results_reverse/other_classifier/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results_reverse/other_classifier/data/mean_test_loss.txt" )

    title = "Precision and loss using neural network with dense layers"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision, color='r' )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision, color='r' )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss, color='r' )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss, color='r' )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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


def plot_less_data():
    NEW_FONT_SIZE = FONT_SIZE - 2

    mean_test_abs_precision = read_file( "thesis_results_reverse/less_data/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results_reverse/less_data/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results_reverse/less_data/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results_reverse/less_data/data/mean_test_loss.txt" )

    title = "Precision and loss using less data"
    subtitles = [ "Absolute precision (a)", "Compatible precision (b)", "Training loss (c)", "Validation loss (d)" ]
    subxtitles = "Training epochs"
    subytitles = [ "Precision", "Cross-entropy loss" ]
    fig, axes = plt.subplots( nrows=2, ncols=2 )
    for row, axis in enumerate( axes ):
        if row == 0:
            # plot top left
            axis[ 0 ].plot( mean_test_abs_precision, color='r' )
            axis[ 0 ].set_title( subtitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 0 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision, color='r' )
            axis[ 1 ].set_title( subtitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )

            for tick in axis[ 0 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].xaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 0 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )
            for tick in axis[ 1 ].yaxis.get_major_ticks():
                tick.label.set_fontsize( NEW_FONT_SIZE )

        if row == 1:
            # plot bottom left
            axis[ 0 ].plot( mean_train_loss, color='r' )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_ylabel( subytitles[ 1 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot bottom right
            axis[ 1 ].plot( mean_val_loss, color='r' )
            axis[ 1 ].set_title( subtitles[ 3 ], fontsize = NEW_FONT_SIZE )
            axis[ 1 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 1 ].grid( True )
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

plot_other_classifier()
#plot_top1_top2_accuracy()
'''plot_activation_perf()
plot_optimiser_perf()
plot_lr_perf()
plot_batchsize_perf()
plot_dropout_perf()
plot_embedding_sizes_perf()
plot_num_units_perf()
plot_extreme_paths()
plot_longer_paths()
plot_train_long_test_decomposed()
plot_top1_top2_accuracy()
plot_input_length_precision()
plot_other_classifier()'''
#plot_tools_compatible_tools( "data/compatible_tools.json" )
#plot_data_distribution( "data/workflow_connections_paths.txt" )
#plot_less_data()
#plot_input_length_precision()
