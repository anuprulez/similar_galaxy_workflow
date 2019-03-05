"""
Predict tool usage to weigh the predicted tools
"""

import sys
import numpy as np
import time
import os
import warnings
import csv

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import utils

warnings.filterwarnings("ignore")


class ToolPopularity:

    @classmethod
    def __init__( self ):
        """ Init method. """
        
    @classmethod
    def extract_tool_usage(self, tool_usage_file):
        """
        Extract the tool usage over time for each tool
        """
        tool_usage_dict = dict()
        tool_list = list()
        with open(tool_usage_file, 'rt') as usage_file:
            tool_usage = csv.reader(usage_file, delimiter='\t')
            for index, row in enumerate(tool_usage):
                tool_id = utils.format_tool_id(row[0])
                tool_list.append(tool_id)
                if tool_id not in tool_usage_dict:
                    tool_usage_dict[tool_id] = dict()
                    tool_usage_dict[tool_id][row[1]] = int(row[2])
                else:
                    curr_date = row[1]
                    if curr_date in tool_usage_dict[tool_id]:
                        tool_usage_dict[tool_id][curr_date] += int(row[2])
                    else:
                        tool_usage_dict[tool_id][curr_date] = int(row[2])
        return tool_usage_dict


    @classmethod
    def learn_tool_popularity(self, x_reshaped, y_reshaped):
        """
        Fit a curve for the tool usage over time to predict future tool usage
        """
        try:
            pipe = Pipeline(steps=[('regressor', ElasticNet())])
            param_grid = {
                'regressor__alpha': [0.1, 0.5, 0.75, 1.0],
            }
            search = GridSearchCV(pipe, param_grid, iid=False, cv=2, return_train_score=False, scoring='r2', n_jobs=2, error_score=1)
            search.fit(x_reshaped, y_reshaped)
            model = search.best_estimator_
            prediction_point = np.reshape([x_reshaped[-1][0] + 1], (1, 1))
            prediction = model.predict(prediction_point)
            if prediction <= 0:
                prediction = [1.0]
            return prediction[0]
        except Exception:
            return 1.0


    @classmethod
    def get_pupularity_prediction(self, tools_usage):
        """
        Get the popularity prediction for each tool
        """
        usage_prediction = dict()
        for tool_name, usage in tools_usage.items():
            y_val = list()
            x_val = list()
            for x, y in usage.items():
                x_val.append(x)
                y_val.append(y)
            x_val = list(reversed(x_val))
            y_val = list(reversed(y_val))
            x_pos = np.arange(len(x_val))
            x_reshaped = x_pos.reshape(len(x_pos), 1)
            y_reshaped = np.reshape(y_val, (len(x_pos), 1))
            prediction = self.learn_tool_popularity(x_reshaped, y_reshaped)
            print(tool_name, prediction)
            usage_prediction[tool_name] = prediction
        utils.write_file("data/generated_files/usage_prediction.txt", usage_prediction)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict_tool_usage.py <tool_usage_file>")
        exit( 1 )
    start_time = time.time()
    
    tool_usage = ToolPopularity()
    usage = tool_usage.extract_tool_usage(sys.argv[1])
    tool_usage.get_pupularity_prediction(usage)
    
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
