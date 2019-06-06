# Predict next tool in Galaxy workflows

## Galaxy workflows as directed graphs
[Galaxy](https://usegalaxy.eu/) workflow is a chain of (Galaxy) tools to process biological data. These datasets undergo a transformation at each node (a tool) which includes text manipulation, sorting on a column, deletion or addition of a column and so on. Each workflow can be considered as a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) where the output of each node becomes an input to the next node(s). Visit this [website](https://rawgit.com/anuprulez/similar_galaxy_workflow/master/viz/index.html) to see all the steps of a workflow and its directed graph. Choose a workflow from the dropdown and see the [Cytoscape](http://js.cytoscape.org/) graph. A typical [workflow](https://usegalaxy.org/workflow/editor?id=4ef668a0f832a731) in Galaxy looks like this:

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/master/images/workflow_galaxy.png">
</p>

## Predict next tool
If a Galaxy user can see a list of possible next tool(s) at all stages of creating a workflow, it would be convenient and time-saving to create one. This work aims to achieve it by training a neural network on the existing set of workflows created by multiple users. There is a special kind of (recurrent) neural network, long short-term memory (LSTM), which learns connections in the input (sequential) data and predicts the next possible connections(s). The Galaxy workflows also qualify as sequential data (as a chain of tools) and this network is expected to work on these data processing pipelines to predict next possible connections. Moreover, this work identifies itself as a part of recommendation system for Galaxy tools and workflows. The approach is explained below:

Suppose we have a workflow:
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/master/images/workflow.png">
</p>

This workflow can be broken down into following smaller sequences (or training samples):

- pileometh > Remove beginning 1 (label)
- pileometh > Remove beginning 1 > Add_a_column1 (label)
- Add_a_column1 > cut1 > addValue (label)
- so on ...

The last item in each such path is a label (or category) of the previous sequence (of tools) which means that this label should be in the predicted set of next tools for that sequence. For example, "pileometh > Remove beginning 1" is a training sample and its label is "Add_a_column1". Following this way, each path is divided into training samples (a tool or a sequence of tools) and their labels. The logic behind this breking up of a workflow is to make the classifier learn that if we are at a stage "pileometh > Remove beginning 1" of creating a workflow, the next tool would be "Add_a_column1". A similar approach is used for predicting next word in sentences. Here, we can draw an analogy between our smaller sequences from workflows and smaller parts of sentences (in English for example). They are similar - sentences in a language like in English (`I → want → to → go → to → Berlin`) and our smaller sequences (`filter1 → grouping1 → addvalue → join1`) as both make sense only when their components are arranged in a particular order.

To feed these input training samples (smaller parts of workflows) into the neural network, we need to convert them into vectors (neural networks understand vectors and not words or text). In order to convert them into vectors, we create a list of unique nodes (tools) and assign them unique integers (let's call them an id for each node). Now, we take a training sample and identify its nodes, take their respective ids and arrange these integers in the same order as the original tool sequence. For example, let's take this small dummy workflow:

`filter1 → grouping1 → addvalue → join1 → add_a_column1`

Let's create a dictionary mapping a unique integer to each tool:

- `{ "addvalue": 1, "add_a_column1": 2, "filter1": 3, "join1": 4, "grouping1": 5 }`

Now, create a training sample - a vector for the workflow:
- `filter1 → grouping1 → addvalue → join1` (training sample)
- `add_a_column1` (a label for the above part of workflow)
- `[ 0, 0,......, 0, 3, 5, 1, 4 ]` (`0`s are added to make up for the maximum length of the input data).

Now, it's time for creating the label vector. It is multi-hot encoded vector which means that this vector is all zeros except for the position(s) of the next tools. For example:

- `[ 0, 0, 1, 0, 0 ]` (a label vector for tool "add_a_column1" because its position value is `2` in the dictionary. So, the `3`rd index of the vector is `1` and others are zeros).

If there are multiple labels for a training sample (which happens to be the case in this work), we add `1s` to all the positions of the corresponding labels).
- `[ 0, 1, 1, 0, 1 ]` shows a multi-hot encoded label vector.

We create training samples and their labels in this manner and feed them to the network. The first layer in the network is an embedding layer which learns a dense, low dimensional vector for each training sample which are largely sparse. These dense, low dimensional vectors are then fed into the LSTM layer. Dropout is added between layers in order to avoid overfitting which happens when the learning (prediction performance) becomes better on training data and stops/saturates on test (unseen) data.

## Data distribution

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/rnn_custom_loss/images/path_dist.png">
</p>

The above plot shows the distribution of length of tool sequences. The length plays an important role to determine the dimensionality of the input dense vector. Thus, to reduce it, we take a maximum tool sequence length of 25.

## Accuracy measure
In the set of training sequences, each one can have many labels (or categories) which means that there can be multiple (next) tools for a sequence of tools. However if we measure accuracy of our approach which predicts just one next tool, it would be partially correct. Hence, we assess the performance on top k predicted tools (top-k accuracy). `20%` of all samples are taken out for testing the trained model's performance and the rest is used to train the model.

## Accuracy on test data

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/rnn_custom_loss/images/precision.png">
</p>

The plot above shows average precision computed over training epochs on test data. The test data makes `20%` of the complete dataset (sequences of tools). 

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/rnn_custom_loss/images/loss.png">
</p>

The plot above shows the binary cross-entropy, unweighted and weighted, loss drop over training epochs. Both the losses, training and validation, start to drop and become stable towards the end of training epochs. 

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/rnn_custom_loss/images/usage.png">
</p>

The plot above shows the increase of mean usage over training epochs. As the precision improves, tools with higher usage are predicted.

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/rnn_custom_loss/images/precision_path_length.png">
</p>

The above plot shows the precision for different length of paths (tool sequences). As the length of path increases, the precision becomes better.

## Literature:
- [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)
- [LSTM by Example using Tensorflow](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)
- [Learning to diagnose with LSTM Recurrent Neural Networks](https://arxiv.org/pdf/1511.03677.pdf)
- [CNN-RNN: A Unified Framework for Multi-label Image Classification](https://arxiv.org/pdf/1604.04573.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Network](https://arxiv.org/pdf/1512.05287.pdf)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/pdf/1412.3555v1.pdf)
- [Fast and Accurate deep network learning by exponential linear units(ELU)](https://arxiv.org/pdf/1511.07289.pdf)
- [Recurrent Neural Network Regularization](https://arxiv.org/pdf/1409.2329.pdf)


## Citations:
Cytoscape.js: a graph theory library for visualisation and analysis
Franz M, Lopes CT, Huck G, Dong Y, Sumer O, Bader GD
Bioinformatics (2016) 32 (2): 309-311 first published online September 28, 2015 doi:10.1093/bioinformatics/btv557 (PDF)
[PubMed Abstract](https://www.ncbi.nlm.nih.gov/pubmed/26415722)
