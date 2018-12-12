import plotly
import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio
import json

# Create random data with numpy
import numpy as np

train_loss_file = "data/generated_files/mean_test_loss.txt"
retrain_loss_file = "data/generated_files/retrain_mean_test_loss.txt"
retrain_loss_no_fine_tuning = "data/generated_files/retrain_no_finetuning_mean_test_loss.txt"

fontsz = 28
circle_size = 10

def read_file(file_name):
    with open(file_name) as fl:
        data = fl.read()
        return data.split("\n")

train_loss = read_file(train_loss_file)
train_loss = train_loss[:len(train_loss) - 1]

retrain_loss = read_file(retrain_loss_file)
retrain_loss = retrain_loss[:len(retrain_loss) - 1]

retrain_loss_no_fine_tuning = read_file(retrain_loss_no_fine_tuning)
retrain_loss_no_fine_tuning = retrain_loss_no_fine_tuning[:len(retrain_loss_no_fine_tuning) - 1]

x_axis = list(range(1, len(train_loss) + 1))

# Create traces
trace0 = go.Scatter(
    x = x_axis,
    y = train_loss,
    mode = 'lines+markers',
    name = 'Training (base)',
    marker = dict(
        size = circle_size,
        color='rgb(243, 1, 1)',
        line = dict(
            width = 2,
            color = 'rgb(243, 1, 1)'
        )
    )
)

trace1 = go.Scatter(
    x = x_axis,
    y = retrain_loss,
    mode = 'lines+markers',
    name = 'Re-training (fine tuning)',
    marker = dict(
        size = circle_size,
        color='rgba(50, 171, 96, 1.0)',
        line = dict(
            width = 2,
            color = 'rgba(50, 171, 96, 1.0)'
        )
    )
)

trace2 = go.Scatter(
    x = x_axis,
    y = retrain_loss_no_fine_tuning,
    mode = 'lines+markers',
    name = 'Re-training (no fine tuning)',
    marker = dict(
        size = circle_size,
        color='rgb(0,191,255)',
        line = dict(
            width = 2,
            color = 'rgb(0,191,255)'
        )
    )
)


layout_time_acc = dict(
    title='Cross-entropy loss comparison for training and re-training with fine tuning',
    font=dict(family='Times new roman', size=fontsz),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        title='Cross-entropy loss',
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='Number of training epochs',
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    margin=dict(
        l=100,
        r=100,
        t=100,
        b=100
    ),
)

fig_tp = go.Figure(data=[trace0, trace1, trace2], layout=layout_time_acc)
pio.write_image(fig_tp, 'data/generated_files/loss_train_retrain.png', width=1200, height=800)

#### Plot precision 

train_precision_file = "data/generated_files/mean_test_absolute_precision.txt"
retrain_precision_file = "data/generated_files/retrain_mean_test_absolute_precision.txt"
retrain_precision_no_fine_tuning = "data/generated_files/retrain_no_finetuning_test_absolute_precision.txt"

train_precision = read_file(train_precision_file)
train_precision = train_precision[:len(train_precision) - 1]

retrain_precision = read_file(retrain_precision_file)
retrain_precision = retrain_precision[:len(retrain_precision) - 1]

retrain_precision_no_fine_tuning = read_file(retrain_precision_no_fine_tuning)
retrain_precision_no_fine_tuning = retrain_precision_no_fine_tuning[:len(retrain_precision_no_fine_tuning) - 1]

# Create traces
trace0 = go.Scatter(
    x = x_axis,
    y = train_precision,
    mode = 'lines+markers',
    name = 'Training (base)',
    marker = dict(
        size = circle_size,
        color='rgb(243, 1, 1)',
        line = dict(
            width = 2,
            color = 'rgb(243, 1, 1)'
        )
    )
)

trace1 = go.Scatter(
    x = x_axis,
    y = retrain_precision,
    mode = 'lines+markers',
    name = 'Re-training (fine tuning)',
    marker = dict(
        size = circle_size,
        color='rgba(50, 171, 96, 1.0)',
        line = dict(
            width = 2,
            color = 'rgba(50, 171, 96, 1.0)'
        )
    )
)

trace2 = go.Scatter(
    x = x_axis,
    y = retrain_precision_no_fine_tuning,
    mode = 'lines+markers',
    name = 'Re-training (no fine tuning)',
    marker = dict(
        size = circle_size,
        color='rgb(0, 191, 255)',
        line = dict(
            width = 2,
            color = 'rgb(0, 191, 255)'
        )
    )
)

layout_time_acc = dict(
    title='Accuracy comparison for training and re-training with fine tuning',
    font=dict(family='Times new roman', size=fontsz),
    yaxis=dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        range=[0.85, 1],
        title='Accuracy',
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    xaxis=dict(
        zeroline=False,
        showline=True,
        showticklabels=True,
        showgrid=True,
        title='Number of training epochs',
        titlefont=dict(
            family='Times new roman',
            size=fontsz
        )
    ),
    margin=dict(
        l=100,
        r=100,
        t=100,
        b=100
    ),
)

fig_tp = go.Figure(data=[trace0, trace1, trace2], layout=layout_time_acc)
pio.write_image(fig_tp, 'data/generated_files/precision_train_retrain.png', width=1200, height=800)

