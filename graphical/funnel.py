from plotly import graph_objects as go

fig = go.Figure()

y = ['Registrations', 'Submitters', 'Projects Submitted']

x_m = [2748, 410, 209]
x_s = [1746, 275, 175]
x_w = [3103, 247, 105]

fig.add_trace(go.Funnel(
    name='Spark AR',
    y=y,
    x=x_s,
    textposition='auto',
    textinfo='value+percent initial'
))

fig.add_trace(go.Funnel(
    name='Messenger Platform',
    y=y,
    x=x_m,
    textposition='inside',
    textinfo='value+percent initial'
))

fig.add_trace(go.Funnel(
    name='Wit.ai',
    y=y,
    x=x_w,
    textposition='outside',
    textinfo='value+percent initial'
))

fig.show()