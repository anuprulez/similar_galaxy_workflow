# Predict next tool in Galaxy workflows

## Galaxy workflows as directed graphs
A Galaxy workflows is a chain of Galaxy tools to process biological datasets. The datasets undergo a transformation at each node (a tool) which includes text manipulation, sorting on a column, deletion or addition of a column and so on. Each workflow can also be assumed as a set of tools arranged as a directed graph where output of each node becomes input to the next nodes. Visit this [website](https://rawgit.com/anuprulez/similar_galaxy_workflow/master/viz/index.html) to see all the steps of a workflow and its graph. Choose a workflow from the dropdown and see its Cytoscape graph.

## Predict next tool
If a Galaxy user can see a list of possible next tool(s) at all stages of creating a workflow, it would be convenient and time-saving to create one. This work aims to achieve it by training a neural network on the existing set of workflows created by multiple users. There is a special kind of (recurrent) neural network, long short-term memory (LSTM), which learns connections in the input (sequential) data and predicts the next possible connections(s). The Galaxy workflows also qualify as sequential data (as a chain of tools) and this network is expected to work on these data processing pipelines to predict next possible connections. Moreover, this work identifies itself as a part of recommendation system for Galaxy tools and workflows. The approach is explained below:

Suppose we have a workflow:
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/workflow1.png">
</p>

This workflow can be broken down into following smaller sequences (or training samples):

- pileometh > Remove beginning 1 (label)
- pileometh > Remove beginning 1 > Add_a_column1 (label)
- Add_a_column1 > cut1 > addValue (label)
- so on ...

The last item in each such paths is the label (or category) of the previous sequence (of tools). For example, "pileometh > Remove beginning 1" is a training sample and its label is "Add_a_column1". Following this way, each path is divided into training samples (a tool or a sequence of tools) and their labels. The logic behind this breking up of a workflow is to make the classifier learn that if we are at a stage "pileometh > Remove beginning 1" of creating a workflow, the next tool would be "Add_a_column1". A similar approach is used for predicting next word in sentences.

To feed the input training samples (smaller parts of workflows) into the neural network, we need to convert them into vectors (neural networks understand vectors and not words or text). Here, we draw an analogy between our smaller sequences from workflows and smaller parts of sentences (in English for example). They are similar - sentences in a language like in English and our smaller sequences as both make sense only when components (tools in our case and words in sentences) are arranged in a particular order. In order to convert them into vectors, we create a list of unique nodes (tools) and assign them unique integers (let's call them an id for each node). Now, we take a training sample and identify its nodes and take the respective ids and arrange these integers in the same order. For example, let's take this small dummy workflow - `filter1 → grouping1 → addvalue → join1 → add_a_column1`

let's create a dictionary:
{ "addvalue": 1, "add_a_column1": 2, "filter1": 3, "join1": 4, "grouping1": 5 }

Now create a training sample - a vector for the workflow:
`filter1 → grouping1 → addvalue → join1` (training sample)
`add_a_column1` (a label for the above part of workflow)

`[ 0, 0,......,0, 3, 5, 1, 4 ]` (0s are added to make up for the maximum length of the input data).

Now, its time for creating the label vector. It is multi-hot encoded vector which means that this vector is all zeros except for the position of the label.
`[ 0, 0, 1, 0, 0 ]` ("add_a_column1" has `2` as its position in the dictionary. So, `2nd` position has `1` and others are zero). If there are multiple labels for a training sample (which happens to be the case in this work, we add `1s` to all the positions of the corresponding labels).

We create training samples and their labels in this manner and feed them to the network.

## Accuracy measure
In our set of training samples, each one can have many labels (or categories) which means that there can be multiple (next) tools for a sequence of tools. However if we measure accuracy of our approach which predicts just one next tool, it would be partially correct. Hence, we assess the performance on top 5 predicted tools (top-5 accuracy). In this accuracy measure, we verify if the actual label(s) is/are present in the top 5 predicted labels for a training sequence.

## Vizualizer

The screenshots below show possible next tools for a sequence at each stage of creating a dummy workflow - `wc_gnu → collapse_dataset → join1 → join1 → cut1 → join1 → filter1 → cut1 → filter1 → grouping1 → addvalue → join1 → add_a_column1 → filter1`

All the paths containing this sequence of tools are also shown.

<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/1.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/2.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/3.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/4.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/5.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/6.png">
</p>
<hr/>
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/7.png">
</p>

In order to run the visualizer, please follow these steps:

1. Download the repository
2. Move to the "viz" folder
3. Install the dependencies (like Keras, Tensorflow, Numpy and h5py)
4. Run the python server: `python viz_server.py 8001`
5. Browse the URL: "http://localhost:8001/"
6. Choose a tool and see the next possible tools
7. Now, choose another tool and so on. At each step of choosing you will find a set of predicted next tools (probability in percentage). 
8. If the given combination is not present, no tools or paths are shown.

Literature:
- [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [A Beginner’s Guide to Recurrent Networks and LSTMs](https://deeplearning4j.org/lstm.html)
- [LSTM by Example using Tensorflow](https://towardsdatascience.com/lstm-by-example-using-tensorflow-feb0c1968537)

Citations:

Cytoscape.js: a graph theory library for visualisation and analysis
Franz M, Lopes CT, Huck G, Dong Y, Sumer O, Bader GD
[Bioinformatics (2016) 32 (2): 309-311 first published online September 28, 2015 doi:10.1093/bioinformatics/btv557 (PDF)](bioinformatics.oxfordjournals.org/content/32/2/309)
[PubMed Abstract](https://www.ncbi.nlm.nih.gov/pubmed/26415722)

<hr/>



