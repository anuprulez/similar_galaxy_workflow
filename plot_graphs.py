
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_data_distribution( file_path ):
    with open( file_path, 'r' ) as dist_file:
        data = json.loads( dist_file.read() )
    freq = [ data[ item ] for item in data ]
    plt.bar( [ index for index, item in enumerate( freq ) ], height=freq, facecolor='g' )
    plt.xlabel( 'Class number (tools that become labels for the sequence)' )
    plt.ylabel( 'Frequency (number of sequences having the same class)' )
    plt.title( 'Workflows: sequence - classes distribution' )
    plt.grid( True )
    plt.show()


def plot_labels_distribution( file_path ):
    with open( file_path, 'r' ) as train_labels:
        labels_distribution = json.loads( train_labels.read() )
    labels_count = list()
    for item in labels_distribution:
        labels_count.append( len( labels_distribution[ item ].split( "," ) ) )
    labels_index = np.arange( len( labels_count ) )
    plt.bar( labels_index, labels_count, facecolor='b', align='center' )
    plt.xlabel( 'Training samples' )
    plt.ylabel( 'Number of labels' )
    plt.title( 'Training samples and their respective number of labels' )
    plt.grid( True )
    plt.show()


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
    plt.legend( [ "train", "test" ] )
    plt.grid( True )
    plt.show()


def plot_accuracy( file_path_train, file_path_test ):
    acc_values_train = list()
    acc_values_test = list()
    with open( file_path_train, 'r' ) as acc_file_train:
        acc_values_train = acc_file_train.read().split( "\n" )
    acc_values_train = [ float( item ) for item in acc_values_train if item ]
    with open( file_path_test, 'r' ) as acc_file_test:
        acc_values_test = acc_file_test.read().split( "\n" )
    acc_values_test = [ float( item ) for item in acc_values_test if item ]   
    plt.plot( acc_values_train )
    plt.plot( acc_values_test )
    plt.ylabel( 'Accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Accuracy vs training epochs' )
    plt.legend( [ "train", "test" ] )
    plt.grid( True )
    plt.show()


def plot_top_prediction( file_path, abs_file_path ):
    loss_values = list()
    with open( file_path, 'r' ) as top_pred:
        pred_values = top_pred.read().split( "\n" )
    with open( abs_file_path, 'r' ) as _abs_top_pred:
        abs_pred_values = _abs_top_pred.read().split( "\n" )
    pred_values = [ float( item ) for item in pred_values if item ]
    abs_pred_values = [ float( item ) for item in abs_pred_values if item ]   
    plt.plot( pred_values )
    plt.plot( abs_pred_values )
    plt.ylabel( 'Accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Next tools (labels) prediction accuracy vs training epochs on test data' )
    plt.legend( [ "Percentage of actual labels in top-5 predicted labels", "Percentage of k actual labels in top-k predicted labels" ] )
    plt.grid( True )
    plt.show()

plot_loss( "data/loss_history.txt", "data/val_loss_history.txt" )
plot_accuracy( "data/accuracy_history.txt", "data/val_accuracy_history.txt" )
plot_top_prediction( "data/top_pred.txt", "data/abs_top_pred.txt" )
#plot_labels_distribution( "data/multi_labels.txt" )
