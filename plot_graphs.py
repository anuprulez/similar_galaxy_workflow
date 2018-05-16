
import json
import matplotlib.pyplot as plt
import numpy as np


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
    print len( comp_labels_index )
    print len( train_seq_count )
    print len( train_labels_count )
    font = { 'family' : 'sans serif', 'size': 22 }
    plt.rc('font', **font) 
    plt.bar( comp_labels_index, test_labels_count, facecolor='r', align='center' )
    plt.xlabel( 'Number of samples' )
    plt.ylabel( 'Number of tools in samples' )
    plt.title( 'Distribution of number of next tools in test samples' )
    plt.grid( True )
    plt.show()

    plt.bar( comp_labels_index, train_labels_count, facecolor='r', align='center' )
    plt.xlabel( 'Number of samples' )
    plt.ylabel( 'Number of labels in samples' )
    plt.title( 'Distribution of number of next tools in test samples' )
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


def plot_accuracy( complete_data_file, test_data_file ):
    acc_values_train = list()
    acc_values_test = list()
    with open( complete_data_file, 'r' ) as acc_complete:
        complete_data_acc = acc_complete.read().split( "\n" )
    complete_data_acc = [ float( item ) for item in complete_data_acc if item ]
    with open( test_data_file, 'r' ) as acc_test:
        test_data_acc = acc_test.read().split( "\n" )    
    test_data_acc = [ float( item ) for item in test_data_acc if item ]
    font = { 'family' : 'sans serif', 'size': 22 }
    plt.rc('font', **font) 
    plt.plot( complete_data_acc )
    plt.plot( test_data_acc )
    plt.ylabel( 'Topk accuracy (0.7 = 70% accuracy)' )
    plt.xlabel( 'Training epochs' )
    plt.title( 'Next tools (labels) pred. topk acc vs. train and test samples' )
    plt.legend( [ "Train samples", "Test samples" ] )
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

#plot_loss( "data/loss_history.txt", "data/val_loss_history.txt" )
#plot_accuracy( "data/abs_top_pred.txt", "data/test_top_pred.txt" )
plot_labels_distribution( "data/test_data_labels_dict.txt", "data/train_data_labels_dict.txt" )

