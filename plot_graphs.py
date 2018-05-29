
import json
import matplotlib.pyplot as plt
import numpy as np
import csv


def plot_data_distribution( file_path ):
    with open( file_path, 'r' ) as dist_file:
        data = json.loads( dist_file.read() )
    freq = [ len( item.split( "," ) ) for item in data ]
    plt.bar( [ index for index, item in enumerate( freq ) ], height=freq, facecolor='g' )
    plt.xlabel( 'Class number (tools that become labels for the sequence)' )
    plt.ylabel( 'Frequency (number of sequences having the same class)' )
    plt.title( 'Workflows: sequence - classes distribution' )
    plt.grid( True )
    plt.show()


def plot_labels_distribution( test_path, train_path ):
    with open( test_path, 'r' ) as test_labels:
        test_labels_distribution = json.loads( test_labels.read() )
    with open( train_path, 'r' ) as train_labels:
        train_labels_distribution = json.loads( train_labels.read() )
    test_labels_count = list()
    test_seq_count = list()
    for item in test_labels_distribution:
        seq = item.split( "," )
        labels = test_labels_distribution[ item ].split( "," )
        test_seq_count.append( len( seq ) )
        test_labels_count.append( len( labels ) )
    
    train_labels_count = list()
    train_seq_count = list()
    for item in train_labels_distribution:
        seq = item.split( "," )
        labels = train_labels_distribution[ item ].split( "," )
        train_seq_count.append( len( seq ) )
        train_labels_count.append( len( labels ) )
    
    train_seq_count.extend( test_seq_count )
    train_labels_count.extend( test_labels_count )
    comp_labels_index = np.arange( len( train_labels_count ) )
    font = { 'family' : 'sans serif', 'size': 22 }
    plt.rc('font', **font) 
    plt.bar( np.arange( len( test_seq_count ) ), test_labels_count, facecolor='r', align='center' )
    plt.xlabel( 'Number of samples' )
    plt.ylabel( 'Number of next compatible tools in samples' )
    plt.title( 'Distribution of number of next tools in test samples' )
    plt.grid( True )
    plt.show()

    '''plt.bar( comp_labels_index, train_labels_count, facecolor='r', align='center' )
    plt.xlabel( 'Number of samples' )
    plt.ylabel( 'Number of next compatible tools in samples' )
    plt.title( 'Distribution of number of next tools in train samples' )
    plt.grid( True )
    plt.show()'''


def plot_loss( file_path_train, file_path_test ):
    loss_values_train = list()
    loss_values_test = list()
    with open( file_path_train, 'r' ) as loss_file_train:
        loss_values_train = loss_file_train.read().split( "\n" )
    loss_values_train = [ float( item ) for item in loss_values_train if item ]
    with open( file_path_test, 'r' ) as loss_file_test:
        loss_values_test = loss_file_test.read().split( "\n" )
    loss_values_test = [ float( item ) for item in loss_values_test if item ]   
    plt.plot( loss_values_train )
    plt.plot( loss_values_test )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epochs' )
    plt.title( 'Loss drop vs epochs' )
    plt.legend( [ "Train data", "Test data" ] )
    plt.grid( True )
    plt.show()


def plot_accuracy( abs_test_file, compatible_test_file ):
    with open( abs_test_file, 'r' ) as abs_test:
        abs_test_acc = abs_test.read().split( "\n" )
    abs_test_acc = [ float( item ) for item in abs_test_acc if item ]
    with open( compatible_test_file, 'r' ) as compatible_test:
        compatible_test_acc = compatible_test.read().split( "\n" )    
    compatible_test_acc = [ float( item ) for item in compatible_test_acc if item ]
    
    #font = { 'family' : 'sans serif', 'size': 22 }
    #plt.rc('font', **font) 
    plt.plot( abs_test_acc )
    plt.plot( compatible_test_acc )
    #plt.plot( abs_train_acc )
    #plt.plot( compatible_train_acc )
    plt.ylabel( 'Topk accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Next tools prediction' )
    plt.legend( [ "Test absolute accuracy", "Test compatible accuracy" ] )
    plt.grid( True )
    plt.show()


def plot_test_accuracy( abs_test_file, compatible_test_file ):
    with open( abs_test_file, 'r' ) as abs_test:
        abs_test_acc = abs_test.read().split( "\n" )
    abs_test_acc = [ float( item ) for item in abs_test_acc if item ]
    with open( compatible_test_file, 'r' ) as compatible_test:
        compatible_test_acc = compatible_test.read().split( "\n" )    
    compatible_test_acc = [ float( item ) for item in compatible_test_acc if item ]
    
    #font = { 'family' : 'sans serif', 'size': 22 }
    #plt.rc('font', **font) 
    plt.plot( abs_test_acc )
    plt.plot( compatible_test_acc )
    plt.ylabel( 'Topk accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Next tools prediction' )
    plt.legend( [ "Test absolute accuracy", "Test compatible accuracy" ] )
    plt.grid( True )
    plt.show()


def plot_top_prediction( abs_file_path ):
    loss_values = list()
    with open( abs_file_path, 'r' ) as _abs_top_pred:
        abs_pred_values = _abs_top_pred.read().split( "\n" )
    abs_pred_values = [ float( item ) for item in abs_pred_values if item ]
    plt.plot( abs_pred_values, marker=".", color="red" )
    plt.ylabel( 'Accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Next tools (labels) prediction accuracy vs training epochs on test data' )
    plt.legend( [ "Percentage of k actual labels in top-k predicted labels" ], loc=2 )
    plt.grid( True )
    plt.show()

def plot_next_tools_precision( file_path ):
    next_tools = list()
    precision = list()
    with open( file_path, 'rb' ) as next_tools_precision:
        test_data_performance = csv.reader( next_tools_precision, delimiter=',' )
        for index, row in enumerate( test_data_performance ):
            tools = row[ 1 ].split(",")
            next_tools.append( len( tools ) )
            precision.append( row[ 6 ] )
    plt.bar( next_tools, precision )
    plt.ylabel( 'Number of next tools' )
    plt.xlabel( 'Precision' )
    plt.title( 'Number of next tool vs precision' )
    plt.grid( True )
    plt.show()

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


def plot_lr():
    lr = 0.001
    decay = 1e-4
    iterations = 200
    lr_rates = list()
    for i in range( iterations ):
        lr_rates.append( lr )
        lr = lr * ( 1 / ( 1 + decay * i ))
    plt.plot( lr_rates )
    plt.show()
 
#plot_lr()
#plot_tools_compatible_tools( "data/compatible_tools.json" )
plot_loss( "data/mean_train_loss.txt", "data/mean_test_loss.txt" )
plot_accuracy( "data/mean_test_absolute_precision.txt", "data/mean_test_compatibility_precision.txt" )
plot_accuracy( "data/mean_test_actual_absolute_precision.txt", "data/mean_test_actual_compatibility_precision.txt" )
#plot_test_accuracy( "data/test_abs_top_pred.txt", "data/test_top_compatible_pred.txt" )
#plot_accuracy( "data/train_abs_top_pred.txt", "data/train_top_compatible_pred.txt" )
#plot_labels_distribution( "data/test_data_labels_dict.txt", "data/train_data_labels_dict.txt" )
#plot_next_tools_precision( "data/test_data_performance_10.csv" )
