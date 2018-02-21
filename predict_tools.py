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
        self.n_input = 5
        # number of units in RNN cell
        self.n_hidden = 256
        self.display_step = 500
        self.training_iters = 50000
        self.raw_paths = list()

    @classmethod
    def process_processed_data( self, fname ):
        tokens = list()
        with open( fname ) as f:
            data = f.readlines()
        self.raw_paths = [ x.replace( "\n", '' ) for x in data ]
        for item in self.raw_paths:
            split_items = item.split( " " )
            for token in split_items:
                if token not in tokens:
                    tokens.append( token )
        tokens = np.array( tokens ) 
        tokens = np.reshape( tokens, [ -1, ] )
        return tokens

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

        dropout = 0.05 #tf.placeholder( tf.float32 )
        layers = 2
        cells = list()
        # reshape to [1, n_input]
        x = tf.reshape( x, [ -1, self.n_input ] )
        x = tf.split( x, self.n_input, 1 )
        for layer in range( layers ):
            cell = rnn.LSTMCell( self.n_hidden )
            cell = rnn.DropoutWrapper( cell, output_keep_prob=1.0 - dropout )
            cells.append( cell )
            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            
            # 2-layer LSTM 
            #cells = rnn.MultiRNNCell( [ rnn.LSTMCell( self.n_hidden ), rnn.LSTMCell( self.n_hidden ) ] )
        cell = rnn.MultiRNNCell( cells )    
        # generate prediction
        outputs, states = rnn.static_rnn( cell, x, dtype=tf.float32 )
        #outputs, states = tf.nn.dynamic_rnn( cell, x, dtype=tf.float32 )

        # there are n_input outputs but
        # we only want the last output
        prediction = tf.matmul(outputs[ -1 ], weights[ 'out' ] ) + biases[ 'out' ]
        return prediction

    @classmethod
    def select_random_path( self ):
        random_number = random.randint( 0, len( self.raw_paths ) - 1 )
        path = self.raw_paths[ random_number ].split(" ")
        if len( path ) > self.n_input + 1:
            return path
        else:
            return self.select_random_path()

    @classmethod
    def execute_rnn(self):
        processed_data = self.process_processed_data( self.train_data )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        vocab_size = len( dictionary )

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

        prediction = self.RNN( x, weights, biases )

        # Loss and optimizer
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2( logits=prediction, labels=y ) )
        #optimizer = tf.train.RMSPropOptimizer( learning_rate=self.learning_rate ).minimize( cost )
        optimizer = tf.train.AdamOptimizer( learning_rate=self.learning_rate ).minimize( cost )

        # Model evaluation
        correct_pred = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )
        accuracy = tf.reduce_mean(tf.cast( correct_pred, tf.float32 ) )

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as session:
            session.run( init )
            step = 0
            acc_total = 0
            loss_total = 0

            while step < self.training_iters:
                # Generate a minibatch. Add some randomness on selection process.
                random_path = self.select_random_path()
                #print(random_path)
                random_start_pos = random.randint( 0, len( random_path ) - self.n_input - 1 )
                symbols_in_keys = [ [ dictionary[ random_path[ i ] ] ] for i in range( random_start_pos, random_start_pos + self.n_input ) ]
                symbols_in_keys = np.reshape( np.array( symbols_in_keys ), [ -1, self.n_input, 1 ] )
                #print(symbols_in_keys)
                symbols_out_onehot = np.zeros( [ vocab_size ], dtype=float )
                #print(random_path[ random_start_pos + self.n_input])
                #print(dictionary[ random_path[ random_start_pos + self.n_input ] ])
                symbols_out_onehot[ dictionary[ random_path[ random_start_pos + self.n_input ] ] ] = 1.0
                #print(symbols_out_onehot)
                symbols_out_onehot = np.reshape( symbols_out_onehot, [ 1,-1 ] )

                #symbols_in_keys, symbols_out_onehot = self.get_random_input_labels( self.n_input )
                _, acc, loss, onehot_pred = session.run( [ optimizer, accuracy, cost, prediction ], \
                                                feed_dict={ x: symbols_in_keys, y: symbols_out_onehot } )
                loss_total += loss
                acc_total += acc
                if ( step + 1 ) % self.display_step == 0:
                    print( "Iter= " + str( step + 1 ) + ", Average Loss= " + \
                        "{:.6f}".format( loss_total / self.display_step ) + ", Average Accuracy= " + \
                        "{:.2f}%".format( 100 * acc_total / self.display_step ) )
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [ random_path[ i ] for i in range( random_start_pos, random_start_pos + self.n_input ) ]
                    symbols_out = random_path[ random_start_pos + self.n_input ]
                    symbols_out_pred = reverse_dictionary[ int( tf.argmax( onehot_pred, 1 ).eval() ) ]
                    print( random_path )
                    print("%s - [%s] vs [%s]" % ( symbols_in, symbols_out, symbols_out_pred ) )
                    print("------------------------------------------")

                    '''print("Random node prediction...")
                    random_input_size = random.randint( 1, self.n_input )
                    random_path = self.select_random_path()
                    random_start_pos = random.randint( 0, len( random_path ) - self.n_input - 1 )
                    symbols_in_keys = [ [ dictionary[ random_path[ i ] ] ] for i in range( random_start_pos, random_start_pos + random_input_size ) ]
                    symbols_in_keys = np.reshape( np.array( symbols_in_keys ), [ -1, random_input_size, 1 ] )
                
                    one_hot_pred = session.run([ prediction ], feed_dict = {x: symbols_in_keys}) 
                    one_hot_pred = np.asarray(one_hot_pred[0]).astype('float64')[0]
                    symbols_in = [ random_path[ i ] for i in range( random_start_pos, random_start_pos + random_input_size ) ]
                    symbols_out = random_path[ random_start_pos + random_input_size ]
                    symbols_out_pred = reverse_dictionary[ int( tf.argmax( one_hot_pred ).eval() ) ]
                    print(random_path)
                    print("%s - [%s] vs [%s]" % ( symbols_in, symbols_out, symbols_out_pred ) )
                    print("=================================")'''
                step += 1

    @classmethod
    def get_random_input_labels( self, input_size, dictionary, vocab_size ):
        random_path = self.select_random_path()
        random_start_pos = random.randint( 0, len( random_path ) - input_size - 1 )
        symbol_in = [ random_path[ i ] for i in range( random_start_pos, random_start_pos + self.n_input ) ]
        symbol_out = random_path[ random_start_pos + self.n_input ]
        symbols_in_keys = [ [ dictionary[ random_path[ i ] ] ] for i in range( random_start_pos, random_start_pos + input_size ) ]
        symbols_in_keys = np.reshape( np.array( symbols_in_keys ), [ -1, input_size, 1 ] )
        symbols_out_onehot = np.zeros( [ vocab_size ], dtype=float )
        symbols_out_onehot[ dictionary[ random_path[ random_start_pos + input_size ] ] ] = 1.0
        symbols_out_onehot = np.reshape( symbols_out_onehot, [ 1,-1 ] )
        return symbols_in_keys, symbols_out_onehot, symbol_in, symbol_out, random_path


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_tools.py" )
        exit( 1 )
    predict_tool = PredictTool( )
    
    predict_tool.execute_rnn()



