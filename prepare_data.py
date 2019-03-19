"""
Prepare the workflow paths to be used by downstream
machine learning algorithm. The paths are divided
into the test and training sets
"""

import os
import collections
import numpy as np
import random

import utils

main_path = os.getcwd()

class PrepareData:

    @classmethod
    def __init__(self, max_seq_length, test_data_share, retrain=False):
        """ Init method. """
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share
        self.retrain = retrain

    @classmethod
    def process_workflow_paths(self, workflow_paths):
        """
        Get all the tools and complete set of individual paths for each workflow
        """
        tokens = list()
        raw_paths = workflow_paths
        raw_paths = [x.replace("\n", '') for x in raw_paths]
        for item in raw_paths:
            split_items = item.split(",")
            for token in split_items:
                if token is not "":
                    tokens.append(token)
        tokens = list(set(tokens))
        tokens = np.array(tokens)
        tokens = np.reshape(tokens, [-1, ])
        return tokens, raw_paths

    @classmethod
    def create_new_dict(self, new_data_dict):
        """
        Create new data dictionary
        """
        reverse_dict = dict((v, k) for k, v in new_data_dict.items())
        return new_data_dict, reverse_dict

    @classmethod
    def assemble_dictionary(self, new_data_dict, old_data_dictionary={}):
        """
        Create/update tools indices in the forward and backward dictionary
        """
        if self.retrain is True or self.retrain is "True":
            dictionary = old_data_dictionary
            max_prev_size = len(dictionary)
            tool_counter = 1
            for tool in new_data_dict:
                if tool not in dictionary:
                    dictionary[tool] = max_prev_size + tool_counter
                    tool_counter += 1
            reverse_dict = dict((v, k) for k, v in dictionary.items())
            return dictionary, reverse_dict
        else:
            new_data_dict, reverse_dict = self.create_new_dict(new_data_dict)
            return new_data_dict, reverse_dict

    @classmethod
    def create_data_dictionary(self, words, old_data_dictionary={}):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter(words).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary) + 1
        dictionary, reverse_dictionary = self.assemble_dictionary(dictionary, old_data_dictionary)
        return dictionary, reverse_dictionary

    @classmethod
    def decompose_paths(self, paths, dictionary):
        """
        Decompose the paths to variable length sub-paths keeping the first tool fixed
        """
        sub_paths_pos = list()
        sub_paths_names = list()
        for index, item in enumerate(paths):
            tools = item.split(",")
            len_tools = len(tools)
            if len_tools <= self.max_tool_sequence_len:
                for window in range(1, len_tools):
                    sequence = tools[0: window + 1]
                    tools_pos = [str(dictionary[str(tool_item)]) for tool_item in sequence]
                    if len(tools_pos) > 1:
                        sub_paths_pos.append(",".join(tools_pos))
                        sub_paths_names.append(",".join(sequence))
        sub_paths_pos = list(set(sub_paths_pos))
        sub_paths_names = list(set(sub_paths_names))
        return sub_paths_pos

    @classmethod
    def prepare_paths_labels_dictionary(self, reverse_dictionary, paths):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths_labels = dict()
        paths_labels_names = dict()
        random.shuffle(paths)
        for item in paths:
            if item and item not in "":
                tools = item.split(",")
                label = tools[-1]
                train_tools = tools[:len(tools) - 1]
                train_tools = ",".join(train_tools)
                if train_tools in paths_labels:
                    paths_labels[train_tools] += "," + label
                else:
                    paths_labels[train_tools] = label
        for item in paths_labels:
            path_names = ",".join([reverse_dictionary[int(pos)] for pos in item.split(",")])
            path_label_names = ",".join([reverse_dictionary[int(pos)] for pos in paths_labels[item].split(",")])
            paths_labels_names[path_names] = path_label_names
        return paths_labels

    @classmethod
    def pad_paths(self, paths_dictionary, num_classes):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len(paths_dictionary)
        data_mat = np.zeros([size_data, self.max_tool_sequence_len])
        label_mat = np.zeros([size_data, num_classes + 1])
        train_counter = 0
        for train_seq, train_label in list(paths_dictionary.items()):
            positions = train_seq.split(",")
            start_pos = self.max_tool_sequence_len - len(positions)
            for id_pos, pos in enumerate(positions):
                data_mat[train_counter][start_pos + id_pos] = int(pos)
            for label_item in train_label.split(","):
                label_mat[train_counter][int(label_item)] = 1.0
            train_counter += 1
        return data_mat, label_mat

    @classmethod
    def split_test_train_data(self, multilabels_paths):
        """
        Split into test and train data randomly for each run
        """
        train_dict = dict()
        test_dict = dict()
        all_paths = multilabels_paths.keys()
        random.shuffle(list(all_paths))
        split_number = int(self.test_share * len(all_paths))
        for index, path in enumerate(list(all_paths)):
            if index < split_number:
                test_dict[path] = multilabels_paths[path]
            else:
                train_dict[path] = multilabels_paths[path]
        return train_dict, test_dict

    @classmethod
    def verify_overlap(self, train_paths, test_paths):
        """
        Verify the overlapping of samples in train and test data
        """
        intersection = list(set(train_paths).intersection(set(test_paths)))
        print("Overlap in train and test: %d" % len(intersection))

    @classmethod
    def get_predicted_usage(self, data_dictionary, predicted_usage):
        """
        Get predicted usage for tools
        """
        usage = dict()
        epsilon = 1.0
        for k, v in data_dictionary.items():
            try:
                usg = predicted_usage[k]
                if usg < epsilon:
                    usg = epsilon
                usage[v] = usg
            except Exception:
                usage[v] = epsilon
                continue
        usage[str(0)] = epsilon
        return usage

    @classmethod
    def assign_class_weights(self, train_labels, predicted_usage):
        """
        Compute class weights using usage
        """
        n_classes = train_labels.shape[1]
        inverted_frequency = dict()
        class_weights = dict()
        epsilon = 1.0
        # get the count of each tool present in the label matrix
        for i in range(1, n_classes):
            count = len(np.where(train_labels[:, i] > 0)[0])
            class_weights[str(i)] = count
        max_frequency = max(class_weights.values())
        for key, frequency in class_weights.items():
            usage = predicted_usage[int(key)]
            if frequency > 0:
                # get inverted frequency for each tool in label matrix
                # to assign higher weight to less frequent tools
                # and lower weight to more frequent tools
                inv_freq = float(max_frequency) / frequency
                if inv_freq < epsilon:
                    inv_freq = epsilon
                inverted_frequency[key] = inv_freq
            else:
                inverted_frequency[key] = epsilon
            # compute combined weight for each tool
            # higher usage, higher weight
            class_weights[key] = np.log(inverted_frequency[key]) + np.log(usage)
        class_weights[str(0)] = epsilon
        inverted_frequency[str(0)] = epsilon
        utils.write_file(main_path + "/data/generated_files/class_weights.txt", class_weights)
        utils.write_file(main_path + "/data/generated_files/inverted_weights.txt", inverted_frequency)
        return class_weights

    @classmethod
    def get_sample_weights(self, train_data, reverse_dictionary, paths_frequency):
        """
        Compute the frequency of paths in training data
        """
        path_weights = np.zeros(len(train_data))
        all_paths = paths_frequency.keys()
        for path_index, path in enumerate(train_data):
            sample = np.reshape(path, (1, len(path)))
            sample_pos = np.where(path > 0)[0]
            sample_tool_pos = path[sample_pos[0]:]
            path_name = ",".join([reverse_dictionary[int(tool_pos)] for tool_pos in sample_tool_pos])
            try:
                path_weights[path_index] = int(paths_frequency[path_name])
            except:
                path_weights[path_index] = 1
        max_path_freq = np.max(path_weights)
        for idx, item in enumerate(path_weights):
            path_weights[idx] = float(max_path_freq) / path_weights[idx]
        return path_weights

    @classmethod
    def get_data_labels_matrices(self, workflow_paths, frequency_paths, old_data_dictionary={}):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_workflow_paths(workflow_paths)
        dictionary, reverse_dictionary = self.create_data_dictionary(processed_data, old_data_dictionary)
        num_classes = len(dictionary)

        print("Raw paths: %d" % len(raw_paths))
        random.shuffle(raw_paths)

        print("Decomposing paths...")
        all_unique_paths = self.decompose_paths(raw_paths, dictionary)
        random.shuffle(all_unique_paths)

        print("Creating dictionaries...")
        multilabels_paths = self.prepare_paths_labels_dictionary(reverse_dictionary, all_unique_paths)

        print("Complete data: %d" % len(multilabels_paths))
        train_paths_dict, test_paths_dict = self.split_test_train_data(multilabels_paths)

        print("Train data: %d" % len(train_paths_dict))
        print("Test data: %d" % len(test_paths_dict))

        utils.write_file(main_path + "/data/generated_files/test_paths_dict.txt", test_paths_dict)
        utils.write_file(main_path + "/data/generated_files/train_paths_dict.txt", train_paths_dict)

        test_data, test_labels = self.pad_paths(test_paths_dict, num_classes)
        train_data, train_labels = self.pad_paths(train_paths_dict, num_classes)
        
        train_sample_weights = self.get_sample_weights(train_data, reverse_dictionary, frequency_paths)

        usage = utils.read_file(main_path + "/data/generated_files/usage_prediction.txt")

        utils.write_file(main_path + "/data/generated_files/data_dict.txt", dictionary)

        # get time decay information
        tool_predicted_usage = self.get_predicted_usage(dictionary, usage)

        # get inverse class weights
        class_weights = self.assign_class_weights(train_labels, tool_predicted_usage)

        return train_data, train_labels, test_data, test_labels, dictionary, reverse_dictionary, class_weights, train_sample_weights
