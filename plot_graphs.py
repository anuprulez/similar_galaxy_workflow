
import matplotlib.pyplot as plt


def plot_loss( file_path ):
    loss_values = list()
    with open( file_path, 'r' ) as loss_file:
        loss_values = loss_file.read().split( "\n" )
    loss_values = [ float( item ) for item in loss_values if item ]   
    plt.plot( loss_values )
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()


def plot_accuracy( file_path ):
    acc_values = list()
    with open( file_path, 'r' ) as acc_file:
        acc_values = acc_file.read().split( "\n" )
    acc_values = [ float( item ) for item in acc_values if item ]   
    plt.plot( acc_values )
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

plot_loss( "data/loss_history.txt" )
plot_accuracy( "data/accuracy_history.txt" )
