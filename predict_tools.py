"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import collections
import time

class PredictTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.train_data = "data/workflow_steps.txt"
        self.learning_rate = 0.001
        self.n_input = 3
        # number of units in RNN cell
        self.n_hidden = 512

    @classmethod
    def process_processed_data( self, fname ):
        with open( fname ) as f:
            content = f.readlines()
        content = [ x.strip() for x in content ]
        content = [content[ i ].split() for i in range( len( content ) ) ]
        content = np.array( content )
        content = np.reshape( content, [ -1, ] )
        return content

    @classmethod
    def create_data_dictionary( self, words ):
        count = collections.Counter( words ).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len( dictionary )  
        reverse_dictionary = dict(zip( dictionary.values(), dictionary.keys() ) )
        return dictionary, reverse_dictionary

    @classmethod
    def RNN( self, x, weights, biases ):

        # reshape to [1, n_input]
        x = tf.reshape( x, [ -1, self.n_input ] )

        # Generate a n_input-element sequence of inputs
        # (eg. [had] [a] [general] -> [20] [6] [33])
        x = tf.split( x, self.n_input, 1 )

        # 2-layer LSTM, each layer has n_hidden units.
        # Average Accuracy= 95.20% at 50k iter
        #rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden),rnn.BasicLSTMCell(self.n_hidden)])

        # 1-layer LSTM with n_hidden units but with lower accuracy.
        # Average Accuracy= 90.60% 50k iter
        # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
        #rnn_cell = rnn.BasicLSTMCell( self.n_hidden )

        # 2-layer LSTM 
        rnn_cell = rnn.MultiRNNCell( [ rnn.LSTMCell( self.n_hidden ), rnn.LSTMCell( self.n_hidden ) ] )
        # generate prediction
        outputs, states = rnn.static_rnn( rnn_cell, x, dtype=tf.float32 )

        # there are n_input outputs but
        # we only want the last output
        return tf.matmul(outputs[ -1 ], weights[ 'out' ] ) + biases[ 'out' ]

    @classmethod
    def execute_rnn(self):

        display_step = 1000
        training_iters = 50000
        processed_data = self.process_processed_data( self.train_data )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        vocab_size = len( dictionary )

        # Parameters
        

        # tf Graph input
        x = tf.placeholder( "float", [ None, self.n_input, 1 ] )
        y = tf.placeholder( "float", [ None, vocab_size ] )

        # RNN output node weights and biases
        weights = {
            'out': tf.Variable( tf.random_normal( [ self.n_hidden, vocab_size ] ) )
        }
        biases = {
            'out': tf.Variable(tf.random_normal( [ vocab_size ] ) )
        }
        pred = self.RNN( x, weights, biases )

        # Loss and optimizer
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( logits=pred, labels=y ) )
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer( learning_rate=self.learning_rate ).minimize( cost )

        # Model evaluation
        correct_pred = tf.equal( tf.argmax( pred, 1 ), tf.argmax( y, 1 ) )
        accuracy = tf.reduce_mean(tf.cast( correct_pred, tf.float32 ) )

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as session:
            session.run( init )
            step = 0
            offset = random.randint( 0, self.n_input + 1 )
            end_offset = self.n_input + 1
            acc_total = 0
            loss_total = 0

            while step < training_iters:
            # Generate a minibatch. Add some randomness on selection process.
                if offset > ( len( processed_data ) - end_offset):
                    offset = random.randint( 0, self.n_input + 1 )
                symbols_in_keys = [ [ dictionary[ str( processed_data[ i ] ) ] ] for i in range( offset, offset + self.n_input ) ]
                symbols_in_keys = np.reshape( np.array( symbols_in_keys ), [ -1, self.n_input, 1 ] )

                symbols_out_onehot = np.zeros( [ vocab_size ], dtype=float )
                symbols_out_onehot[ dictionary[ str( processed_data[ offset + self.n_input ] ) ] ] = 1.0
                symbols_out_onehot = np.reshape( symbols_out_onehot,[ 1,-1 ] )

                _, acc, loss, onehot_pred = session.run( [ optimizer, accuracy, cost, pred ], \
                                                feed_dict={ x: symbols_in_keys, y: symbols_out_onehot } )
                loss_total += loss
                acc_total += acc
                if ( step + 1 ) % display_step == 0:
                    print( "Iter= " + str( step + 1 ) + ", Average Loss= " + \
                        "{:.6f}".format( loss_total / display_step ) + ", Average Accuracy= " + \
                        "{:.2f}%".format( 100 * acc_total / display_step ) )
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [ processed_data[ i ] for i in range( offset, offset + self.n_input ) ]
                    symbols_out = processed_data[ offset + self.n_input ]
                    symbols_out_pred = reverse_dictionary[ int( tf.argmax( onehot_pred, 1 ).eval() ) ]
                    print("%s - [%s] vs [%s]" % ( symbols_in, symbols_out, symbols_out_pred ) )
                step += 1
                offset += ( self.n_input + 1 )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_tools.py" )
        exit( 1 )
    predict_tool = PredictTool()
    predict_tool.execute_rnn()



