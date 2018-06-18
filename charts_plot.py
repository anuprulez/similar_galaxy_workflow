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
    plt.bar( np.arange( len( next_tools ) ), next_tools, color='r' )
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

    mean_test_abs_precision_elu = read_file( "thesis_results/activations/elu/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_elu = read_file( "thesis_results/activations/elu/mean_test_compatibility_precision.txt" )
    mean_train_loss_elu = read_file( "thesis_results/activations/elu/mean_train_loss.txt" )
    mean_val_loss_precision_elu = read_file( "thesis_results/activations/elu/mean_test_loss.txt" )
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
            axis[ 0 ].plot( mean_test_abs_precision_relu )
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
    NEW_FONT_SIZE = FONT_SIZE - 6
    mean_test_abs_precision_sgd = read_file( "thesis_results/optimiser/sgd/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_sgd = read_file( "thesis_results/optimiser/sgd/mean_test_compatibility_precision.txt" )
    mean_train_loss_sgd = read_file( "thesis_results/optimiser/sgd/mean_train_loss.txt" )
    mean_val_loss_sgd = read_file( "thesis_results/optimiser/sgd/mean_test_loss.txt" )

    mean_test_abs_precision_adagrad = read_file( "thesis_results/optimiser/adagrad/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_adagrad = read_file( "thesis_results/optimiser/adagrad/mean_test_compatibility_precision.txt" )
    mean_train_loss_adagrad = read_file( "thesis_results/optimiser/adagrad/mean_train_loss.txt" )
    mean_val_loss_adagrad = read_file( "thesis_results/optimiser/adagrad/mean_test_loss.txt" )

    mean_test_abs_precision_adam = read_file( "thesis_results/optimiser/adam/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_adam = read_file( "thesis_results/optimiser/adam/mean_test_compatibility_precision.txt" )
    mean_train_loss_adam = read_file( "thesis_results/optimiser/adam/mean_train_loss.txt" )
    mean_val_loss_adam = read_file( "thesis_results/optimiser/adam/mean_test_loss.txt" )

    mean_test_abs_precision_rmsprop = read_file( "thesis_results/optimiser/rmsprop/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_rmsprop = read_file( "thesis_results/optimiser/rmsprop/mean_test_compatibility_precision.txt" )
    mean_train_loss_rmsprop = read_file( "thesis_results/optimiser/rmsprop/mean_train_loss.txt" )
    mean_val_loss_precision_rmsprop = read_file( "thesis_results/optimiser/rmsprop/mean_test_loss.txt" )
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
    NEW_FONT_SIZE = FONT_SIZE - 6
    mean_test_abs_precision_01 = read_file( "thesis_results/lr/0.01/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_01 = read_file( "thesis_results/lr/0.01/mean_test_compatibility_precision.txt" )
    mean_train_loss_01 = read_file( "thesis_results/lr/0.01/mean_train_loss.txt" )
    mean_val_loss_01 = read_file( "thesis_results/lr/0.01/mean_test_loss.txt" )

    mean_test_abs_precision_005 = read_file( "thesis_results/lr/0.005/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_005 = read_file( "thesis_results/lr/0.005/mean_test_compatibility_precision.txt" )
    mean_train_loss_005 = read_file( "thesis_results/lr/0.005/mean_train_loss.txt" )
    mean_val_loss_005 = read_file( "thesis_results/lr/0.005/mean_test_loss.txt" )

    mean_test_abs_precision_001 = read_file( "thesis_results/lr/0.001/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_001 = read_file( "thesis_results/lr/0.001/mean_test_compatibility_precision.txt" )
    mean_train_loss_001 = read_file( "thesis_results/lr/0.001/mean_train_loss.txt" )
    mean_val_loss_001 = read_file( "thesis_results/lr/0.001/mean_test_loss.txt" )

    mean_test_abs_precision_0001 = read_file( "thesis_results/lr/0.0001/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_0001 = read_file( "thesis_results/lr/0.0001/mean_test_compatibility_precision.txt" )
    mean_train_loss_0001 = read_file( "thesis_results/lr/0.0001/mean_train_loss.txt" )
    mean_val_loss_precision_0001 = read_file( "thesis_results/lr/0.0001/mean_test_loss.txt" )
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
    NEW_FONT_SIZE = FONT_SIZE - 6
    mean_test_abs_precision_64 = read_file( "thesis_results/batchsize/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results/batchsize/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results/batchsize/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results/batchsize/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results/batchsize/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results/batchsize/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results/batchsize/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results/batchsize/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results/batchsize/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results/batchsize/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results/batchsize/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results/batchsize/256/mean_test_loss.txt" )

    mean_test_abs_precision_512 = read_file( "thesis_results/batchsize/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results/batchsize/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results/batchsize/512/mean_train_loss.txt" )
    mean_val_loss_precision_512 = read_file( "thesis_results/batchsize/512/mean_test_loss.txt" )
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
    NEW_FONT_SIZE = FONT_SIZE - 6
    mean_test_abs_precision_00 = read_file( "thesis_results/dropout/0.0/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_00 = read_file( "thesis_results/dropout/0.0/mean_test_compatibility_precision.txt" )
    mean_train_loss_00 = read_file( "thesis_results/dropout/0.0/mean_train_loss.txt" )
    mean_val_loss_00 = read_file( "thesis_results/dropout/0.0/mean_test_loss.txt" )

    mean_test_abs_precision_01 = read_file( "thesis_results/dropout/0.1/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_01 = read_file( "thesis_results/dropout/0.1/mean_test_compatibility_precision.txt" )
    mean_train_loss_01 = read_file( "thesis_results/dropout/0.1/mean_train_loss.txt" )
    mean_val_loss_01 = read_file( "thesis_results/dropout/0.1/mean_test_loss.txt" )

    mean_test_abs_precision_02 = read_file( "thesis_results/dropout/0.2/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_02 = read_file( "thesis_results/dropout/0.2/mean_test_compatibility_precision.txt" )
    mean_train_loss_02 = read_file( "thesis_results/dropout/0.2/mean_train_loss.txt" )
    mean_val_loss_02 = read_file( "thesis_results/dropout/0.2/mean_test_loss.txt" )

    mean_test_abs_precision_03 = read_file( "thesis_results/dropout/0.3/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_03 = read_file( "thesis_results/dropout/0.3/mean_test_compatibility_precision.txt" )
    mean_train_loss_03 = read_file( "thesis_results/dropout/0.3/mean_train_loss.txt" )
    mean_val_loss_03 = read_file( "thesis_results/dropout/0.3/mean_test_loss.txt" )

    mean_test_abs_precision_04 = read_file( "thesis_results/dropout/0.4/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_04 = read_file( "thesis_results/dropout/0.4/mean_test_compatibility_precision.txt" )
    mean_train_loss_04 = read_file( "thesis_results/dropout/0.4/mean_train_loss.txt" )
    mean_val_loss_04 = read_file( "thesis_results/dropout/0.4/mean_test_loss.txt" )

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
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].grid( True )

            # plot top right
            axis[ 1 ].plot( mean_test_comp_precision_00 )
            axis[ 1 ].plot( mean_test_comp_precision_01 )
            axis[ 1 ].plot( mean_test_comp_precision_02 )
            axis[ 1 ].plot( mean_test_comp_precision_03 )
            axis[ 1 ].plot( mean_test_comp_precision_04 )
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
            axis[ 0 ].plot( mean_train_loss_00 )
            axis[ 0 ].plot( mean_train_loss_01 )
            axis[ 0 ].plot( mean_train_loss_02 )
            axis[ 0 ].plot( mean_train_loss_03 )
            axis[ 0 ].plot( mean_train_loss_04 )
            axis[ 0 ].set_title( subtitles[ 2 ], fontsize = NEW_FONT_SIZE )
            axis[ 0 ].set_xlabel( subxtitles, fontsize = NEW_FONT_SIZE )
            axis[ 0 ].legend( legend, fontsize = NEW_FONT_SIZE )
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


def plot_embedding_sizes_perf():
    NEW_FONT_SIZE = FONT_SIZE - 6

    mean_test_abs_precision_32 = read_file( "thesis_results/embeddinglayersize/32/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_32 = read_file( "thesis_results/embeddinglayersize/32/mean_test_compatibility_precision.txt" )
    mean_train_loss_32 = read_file( "thesis_results/embeddinglayersize/32/mean_train_loss.txt" )
    mean_val_loss_32 = read_file( "thesis_results/embeddinglayersize/32/mean_test_loss.txt" )

    mean_test_abs_precision_64 = read_file( "thesis_results/embeddinglayersize/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results/embeddinglayersize/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results/embeddinglayersize/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results/embeddinglayersize/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results/embeddinglayersize/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results/embeddinglayersize/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results/embeddinglayersize/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results/embeddinglayersize/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results/embeddinglayersize/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results/embeddinglayersize/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results/embeddinglayersize/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results/embeddinglayersize/256/mean_test_loss.txt" )

    mean_test_abs_precision_512 = read_file( "thesis_results/embeddinglayersize/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results/embeddinglayersize/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results/embeddinglayersize/512/mean_train_loss.txt" )
    mean_val_loss_512 = read_file( "thesis_results/embeddinglayersize/512/mean_test_loss.txt" )

    mean_test_abs_precision_1024 = read_file( "thesis_results/embeddinglayersize/1024/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_1024 = read_file( "thesis_results/embeddinglayersize/1024/mean_test_compatibility_precision.txt" )
    mean_train_loss_1024 = read_file( "thesis_results/embeddinglayersize/1024/mean_train_loss.txt" )
    mean_val_loss_1024 = read_file( "thesis_results/embeddinglayersize/1024/mean_test_loss.txt" )

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
    NEW_FONT_SIZE = FONT_SIZE - 6

    mean_test_abs_precision_64 = read_file( "thesis_results/#units/64/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_64 = read_file( "thesis_results/#units/64/mean_test_compatibility_precision.txt" )
    mean_train_loss_64 = read_file( "thesis_results/#units/64/mean_train_loss.txt" )
    mean_val_loss_64 = read_file( "thesis_results/#units/64/mean_test_loss.txt" )

    mean_test_abs_precision_128 = read_file( "thesis_results/#units/128/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_128 = read_file( "thesis_results/#units/128/mean_test_compatibility_precision.txt" )
    mean_train_loss_128 = read_file( "thesis_results/#units/128/mean_train_loss.txt" )
    mean_val_loss_128 = read_file( "thesis_results/#units/128/mean_test_loss.txt" )

    mean_test_abs_precision_256 = read_file( "thesis_results/#units/256/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_256 = read_file( "thesis_results/#units/256/mean_test_compatibility_precision.txt" )
    mean_train_loss_256 = read_file( "thesis_results/#units/256/mean_train_loss.txt" )
    mean_val_loss_256 = read_file( "thesis_results/#units/256/mean_test_loss.txt" )
    
    mean_test_abs_precision_512 = read_file( "thesis_results/#units/512/mean_test_absolute_precision.txt" )
    mean_test_comp_precision_512 = read_file( "thesis_results/#units/512/mean_test_compatibility_precision.txt" )
    mean_train_loss_512 = read_file( "thesis_results/#units/512/mean_train_loss.txt" )
    mean_val_loss_512 = read_file( "thesis_results/#units/512/mean_test_loss.txt" )

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
    NEW_FONT_SIZE = FONT_SIZE - 6

    mean_test_abs_precision = read_file( "thesis_results/extreme_paths/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results/extreme_paths/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results/extreme_paths/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results/extreme_paths/data/mean_test_loss.txt" )

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
    NEW_FONT_SIZE = FONT_SIZE - 6

    mean_test_abs_precision = read_file( "thesis_results/longer_train_paths/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results/longer_train_paths/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results/longer_train_paths/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results/longer_train_paths/data/mean_test_loss.txt" )

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
    NEW_FONT_SIZE = FONT_SIZE - 6

    mean_test_abs_precision = read_file( "thesis_results/train_long_test_decomposed/data/mean_test_absolute_precision.txt" )
    mean_test_comp_precision = read_file( "thesis_results/train_long_test_decomposed/data/mean_test_compatibility_precision.txt" )
    mean_train_loss = read_file( "thesis_results/train_long_test_decomposed/data/mean_train_loss.txt" )
    mean_val_loss = read_file( "thesis_results/train_long_test_decomposed/data/mean_test_loss.txt" )

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

#plot_activation_perf()
#plot_optimiser_perf()
#plot_lr_perf()
#plot_batchsize_perf()
plot_dropout_perf()
#plot_embedding_sizes_perf()
#plot_num_units_perf()
#plot_extreme_paths()
#plot_longer_paths()
#plot_train_long_test_decomposed()
plot_tools_compatible_tools( "data/compatible_tools.json" )
#plot_data_distribution( "data/workflow_connections_paths.txt" )
