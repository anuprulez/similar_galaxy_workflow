 @classmethod
    def learn_graph_vector( self, tagged_documents ):
        """
        Learn a vector representation for each graph
        """   
        training_epochs = 20
        fix_graph_dimension = 100
        len_graphs = len( tagged_documents )
        print ('Learning doc2vectors...')
        input_vector = np.zeros( [ len_graphs, fix_graph_dimension ] )
        model = gensim.models.Doc2Vec( tagged_documents, dm=0, size=fix_graph_dimension, negative=5, min_count=1, iter=200, window=15, alpha=1e-2, min_alpha=1e-4, dbow_words=1, sample=1e-5 )
        for epoch in range( training_epochs ):
            print ( 'Learning vector repr. epoch %s' % epoch )
            shuffle( tagged_documents )
            model.train( tagged_documents, total_examples=model.corpus_count, epochs=model.iter )
        for i in range( len( model.docvecs ) ):
           input_vector[ i ][ : ] = model.docvecs[ i ]
        with h5.File( self.doc2vec_model_path, "w" ) as model_file:
            model_file.create_dataset( "doc2vector", input_vector.shape, data=input_vector, dtype='float64' )
        return input_vector

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        test_data_share = 0.33
        seed = 0
        data = prepare_data.PrepareData()
        complete_data, labels, dictionary, reverse_dictionary, tagged_documents = data.read_data()
        print ("Learning vector representations of graphs...")
        complete_data_vector = self.learn_graph_vector( tagged_documents )
        np.random.seed( seed )
        dimensions = len( dictionary )
        train_data, test_data, train_labels, test_labels = train_test_split( complete_data_vector, labels, test_size=test_data_share, random_state=seed )
        # write the test data and labels to files for further evaluation
        with h5.File( self.test_data_path, "w" ) as test_data_file:
            test_data_file.create_dataset( "testdata", test_data.shape, data=test_data )
        with h5.File( self.test_labels_path, "w" ) as test_labels_file:
            test_labels_file.create_dataset( "testlabels", test_labels.shape, data=test_labels )
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print ("Dividing data...")
        n_epochs = 700
        num_predictions = 5
        batch_size = 40
        dropout = 0.2
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary = self.divide_train_test_data()
        # reshape train and test data
        train_data = np.reshape( train_data, ( train_data.shape[0], 1, train_data.shape[1] ) )
        train_labels = np.reshape( train_labels, (train_labels.shape[0], 1, train_labels.shape[1] ) )
        test_data = np.reshape(test_data, ( test_data.shape[0], 1, test_data.shape[1] ) )
        test_labels = np.reshape( test_labels, ( test_labels.shape[0], 1, test_labels.shape[1] ) )
        train_data_shape = train_data.shape
        optimizer = Adam( lr=0.0001 )
        # define recurrent network
        model = Sequential()
        model.add( LSTM( 256, input_shape=( train_data_shape[ 1 ], train_data_shape[ 2 ] ), return_sequences=True, recurrent_dropout=dropout ) )
        model.add( Dropout( dropout ) )
        #model.add( LSTM( 512, return_sequences=True, recurrent_dropout=dropout ) )
        #model.add( Dropout( dropout ) )
        model.add( LSTM( 256, return_sequences=True, recurrent_dropout=dropout ) )
        model.add( Dense( 256 ) )
        model.add( Dropout( dropout ) )
        model.add( Dense( dimensions ) )
        model.add( Activation( 'softmax' ) )
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=[ 'accuracy' ] )

        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        callbacks_list = [ checkpoint ]

        print ("Start training...")
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        accuracy_values = model_fit_callbacks.history[ "acc" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        validation_acc = model_fit_callbacks.history[ "val_acc" ]

        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.accuracy_path, np.array( accuracy_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        np.savetxt( self.val_accuracy_path, np.array( validation_acc ), delimiter="," )

        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write(model_json)
        # save the learned weights to h5 file
        model.save_weights( self.weights_path )
        print ("Training finished")

