import numpy as np

import prepare_data
import evaluate_top_results


def evaluate_results():
    n_epochs = 1000
    num_predictions = 5
    data = prepare_data.PrepareData()
    complete_data, labels, dictionary, reverse_dictionary = data.read_data()
    complete_data = complete_data[ :len( complete_data ) - 1 ]
    labels = labels[ :len( labels ) - 1 ]
    len_data = len( complete_data )
    dimensions = len( complete_data[ 0 ] )

    # create test and train data and labels
    complete_data_reshaped = np.zeros( [ len_data, dimensions ] )
    labels_reshaped = np.zeros( [ len_data, dimensions ] )

    for i, item in enumerate( complete_data ):
        complete_data_reshaped[ i ] = complete_data[ i ]
        labels_reshaped[ i ] = labels[ i ]

    data = np.reshape( complete_data_reshaped, ( complete_data_reshaped.shape[ 0 ], 1, complete_data_reshaped.shape[ 1 ] ) )
    labels = np.reshape( labels_reshaped, ( labels_reshaped.shape[ 0 ], 1, labels_reshaped.shape[ 1 ] ) )

    predict_tool = evaluate_top_results.EvaluateTopResults()
    predict_tool.evaluate_topn_epochs( n_epochs, num_predictions, dimensions, reverse_dictionary, data, labels )

evaluate_results()
