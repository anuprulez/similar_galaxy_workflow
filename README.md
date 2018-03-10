# Predict next tool in Galaxy workflows

## Galaxy workflows as directed graphs
The Galaxy workflows are directed graphs in which each node is represented by a Galaxy tool. It is a data processing pipeline through which datasets undergo some transformation at each node. These transformations include text manipulation, sorting on a column, deletion or addition of a column and so on.
Each workflow consists of a certain number of tools arranged as a directed graph. Visit this [website](https://rawgit.com/anuprulez/similar_galaxy_workflow/master/viz/index.html) to see all the steps of a workflow and its graph.

## Predict next tool
If a user can see a list of possible next tool(s) at any stage of creating workflows, it would be convenient and time-saving. This work tries to achieve it by training a neural network (deep learning) on the existing set of workflows from multiple users. It learns patterns/connections from the workflows among tools following a classification approach. The approach is explained below:

Suppose we have a workflow:
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/workflow1.png">
</p>

This workflow can be broken down into following smaller sequences (or training samples):

- pileometh > Remove beginning 1 (label)
- pileometh > Remove beginning 1 > Add_a_column1 (label)
- Add_a_column1 > cut1 > addValue (label)
- so on ...

The last item in each such paths is the label (or category) of the previous sequence (of tools). For example, "pileometh > Remove beginning 1" is a training sample and its label is "Add_a_column1". Following this way, each path is divided into training samples (a tool or a sequence of tools) and their labels. The logic behind this breaking up of a workflow is to make the classifier learn that if we are at a stage "pileometh > Remove beginning 1" of creating a workflow, the next tool would be "Add_a_column1". A similar approach is used for predicting next word in sentences.

To feed the input training samples (smaller parts of workflows) into the neural network, we need to convert them into vectors. Here, we draw an analogy between our smaller sequences from workflows and smaller parts of sentences (in English for example). They are similar - sentences in a language like in English and our smaller sequences as both make sense only when components (tools in our case and words in sentences) are arranged in a particular order. Following it, we translate our training sequences as vectors. Each training sample becomes a vector after training them on another neural network ([doc2vec](https://cs.stanford.edu/%7Equocle/paragraph_vector.pdf)). These vectors are fed into a dense neural network while the labels are embedded as one-hot vector.

## Accuracy measure
In our set of training samples, each one can have many labels (or categories) which means that there can be multiple (next) tools for a sequence of tools. However if we measure accuracy of our approach which predicts just one next tool, it would be partially correct. Hence, we assess the performance on top 5 predicted tools (top-5 accuracy). In this accuracy measure, we verify if the actual label is present in the top 5 predicted labels for a training sequence.

## Visualizer
<p align="center">
  <img src="https://raw.githubusercontent.com/anuprulez/similar_galaxy_workflow/doc2vec_tools_sequences/images/predict-next-tools.png">
</p>
The above image shows possible next tools for a sequence. All the paths containing this sequence of tools are also shown.
In order to run the visualizer, please follow these steps:

1. Download the repository
2. Move to the "viz" folder
3. Install the dependencies (like [Keras](https://keras.io/), [Tensorflow](https://www.tensorflow.org/), Numpy and [h5py](https://www.h5py.org/)). Please install if there are more dependencies. 
4. Run the python server: `python viz_server.py 8001`
5. Browse the URL: "http://localhost:8001/"
6. Choose a tool and see the next possible tools
7. Now, choose another tool and so on. At each step of choosing you will find a set of predicted next tools. 
8. If the given combination is not present, no tools or paths are shown.

## How to use the script
1. First of all, extract the archived workflows `data/workflows.tar.gz` in the same folder. It will result in `workflows` directory containing sub-directories of workflows.
2. Execute `python extract_workflows.py` which would take ~30 seconds to create data files for unique tool names and json files for all workflows. For example, `data/workflows.json`, `data/all_tools.csv` and `data/processed_workflows.csv`
3. Next, execute `python predict_next_tool.py`. This will create uniques paths of tools (as in directed graphs) from all the extracted workflows. After this, it will learn individual vectors (100 dimensions) for each path. These vectors are fed to a neural network for training. Once the execution finishes, we would have `h5` files containing learned weight vectors for each training epoch.
4. It is time for evaluating our learned model. Execute `python evaluate_top_results.py` to learn the top-5 accuracy (by default). Please change a variable `num_predictions` to have top-n accuracy.
5. To plot graphs for loss drop or accuracy increase, run `python plot_graphs.py`.

## Citations:

@inproceedings{rehurek_lrec,
      title = {{Software Framework for Topic Modelling with Large Corpora}},
      author = {Radim {\v R}eh{\r u}{\v r}ek and Petr Sojka},
      booktitle = {{Proceedings of the LREC 2010 Workshop on New
           Challenges for NLP Frameworks}},
      pages = {45--50},
      year = 2010,
      month = May,
      day = 22,
      publisher = {ELRA},
      address = {Valletta, Malta},
      note={\url{http://is.muni.cz/publication/884893/en}},
      language={English}
}
