
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


def plot_loss( file_path ):
    loss_values = list()
    with open( file_path, 'r' ) as loss_file:
        loss_values = loss_file.read().split( "\n" )
    loss_values = [ float( item ) for item in loss_values if item ]   
    plt.plot( loss_values )
    plt.ylabel( 'Loss' )
    plt.xlabel( 'Epochs' )
    plt.grid( True )
    plt.show()


def plot_accuracy( file_path ):
    acc_values = list()
    with open( file_path, 'r' ) as acc_file:
        acc_values = acc_file.read().split( "\n" )
    acc_values = [ float( item ) for item in acc_values if item ]   
    plt.plot( acc_values )
    plt.ylabel( 'Accuracy' )
    plt.xlabel( 'Epochs' )
    plt.grid( True )
    plt.show()


plot_data_distribution( "data/data_distribution.txt" ) 
plot_loss( "data/loss_history.txt" )
plot_accuracy( "data/accuracy_history.txt" )
