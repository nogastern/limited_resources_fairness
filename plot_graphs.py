import plotly.graph_objects as go
import os


def max_min_by_iteration(max_min_values, num_of_iters, sensitive_feature, filename_prefix):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(num_of_iters)), y=max_min_values, mode='lines'))

    fig.update_layout(
        title=f'Post-Processing - {sensitive_feature}',
        xaxis_title='Iteration',
        yaxis_title='Max-Min',
        yaxis_range=[0, 1]
    )
    filename = f'{filename_prefix}_max_min_post_processing_{sensitive_feature}_graph.html'
    fig.write_html(filename)

    # Print the path to the HTML file
    print(f'Graph saved: {os.path.abspath(filename)}')


def metric_by_iteration(data, metric, feature, num_of_iters, filename_prefix=''):
    fig = go.Figure()

    for group, values in data.items():
        fig.add_trace(go.Scatter(x=list(range(num_of_iters)), y=values, mode='lines', name=group))

    fig.update_layout(
        title=f'Post-Processing - {feature}',
        xaxis_title='Iteration',
        yaxis_title=metric,
        yaxis_range=[0, 1]
    )
    filename = f'{filename_prefix}_post_processing_{feature}_{metric}_graph.html'
    fig.write_html(filename)

    # Print the path to the HTML file
    print(f'Graph saved: {os.path.abspath(filename)}')
