import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.misc import derivative
from inspect import getfullargspec

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from dash import no_update, callback_context

import dash

app = dash.Dash(__name__)


def get_array(start, end, step):
    """np.linspace() to avoid array shape errors (due to float precision)
        but with np.arange() signature for usability
    Args:
        start, end, step: float (end is included in array)
    Returns:
        ndarray
    """
    num = int(np.rint((end - start) / step))
    return np.linspace(start, end, num + 1)


def integrate(t_array, diff_eq, y_ini=0, method="LSODA"):
    """Integrate differential equation over t_array with initial value stock_ini.
    Args:
        t_array (ndarray)
        y_ini (int / ndarray)
        diff_eq: function with time and stock as parameter
        method:
            - 'LSODA': for fast result (default)
            - 'RK45', 'RK23', 'DOP853' for non-stiff problems (use solve_ivp)
            - 'Radau', 'BDF' for stiff problem (use solve_ivp)
    Returns:
        ndarray
    """

    # check number argument of diff_eq
    nb_args_diff_eq = len(getfullargspec(diff_eq).args)

    if nb_args_diff_eq == 1:

        def function(t, y):
            return diff_eq(y)

    elif nb_args_diff_eq == 2:
        function = diff_eq
    else:
        raise Exception(
            "diff_eq only accepts 1 argument (autonomous) or 2 arguments (non-autonomous)"
        )

    with np.errstate(all="warn"):
        y = solve_ivp(
            function,
            (t[0], t[-1]),
            # solve_ivp requires y0 to be an ndarray
            np.atleast_1d(y_ini),
            method=method,
            t_eval=t,
        ).y  # error tolerance (default= 1e-3 can diverge with RK45)

    return np.squeeze(y)


# Parameters
t = get_array(start=0, end=10, step=0.01)
x_ini = 1


# Differential equation
def diff_eq(x):
    dx_dt = np.sin(x)
    return dx_dt


# set up slider
myslider = dcc.Slider(
    id="myslider",
    min=0,
    max=2.25,
    step=0.05,
    value=0.25,
    updatemode="drag",
    marks={int(i): {} for i in range(0, 3)}
    | {
        float(i): {
            "label": str(i) + "\U0001D70B",
            "style": {"font-size": "large", "font-weight": "bold"},
        }
        for i in np.arange(0, 2.3, 0.25)
    },
)


# set up dropdown (solver options)
methods = []
for item in ["LSODA", "RK45", "RK23", "DOP853", "Radau", "BDF"]:
    methods.append({"label": item, "value": item})
solver_options = dcc.Dropdown(id="solver", options=methods, value="LSODA")


# Set up Phase Portrait graph
y = get_array(start=-7, end=7, step=0.1)
dy_dt = diff_eq(y)
y_equals_0 = go.Scatter(
    x=y, y=np.zeros(len(y)), mode="lines", name="y = 0", line=dict(color="black")
)
differential = go.Scatter(
    x=y, y=dy_dt, mode="lines", name="differential", line=dict(color="blue")
)
stable_points = go.Scatter(
    x=[],
    y=[],
    mode="markers",
    name="stable",
    marker=dict(size=12, color="black", line={"width": 2}),
)
unstable_points = go.Scatter(
    x=[],
    y=[],
    mode="markers",
    name="unstable",
    marker=dict(size=12, color="white", line={"width": 2}),
)
data = [y_equals_0, differential, stable_points, unstable_points]
layout = go.Layout(title="Phase Portrait", hovermode="closest", template="plotly_white")
phase_portrait = dcc.Graph(id="phase_portrait", figure={"data": data, "layout": layout})


# Set up Dynamics graph
mygraph = dcc.Graph(id="graph")


# Set up app layout
app.layout = html.Div(
    children=[
        html.Div(
            children=[
                dcc.Store(id="stored_stable_fixed_pts", data=[]),
                html.Div(
                    id="printed_stable_fixed_pts", style={"display": "inline-block"}
                ),
                dcc.Store(id="stored_unstable_fixed_pts", data=[]),
                html.Div(
                    id="printed_unstable_fixed_pts", style={"display": "inline-block"}
                ),
            ],
            id="fixed_points",
            style={"width": "70%", "margin": "0 auto"},
        ),
        html.Div(phase_portrait, style={"width": "70%", "margin": "0 auto"}),
        html.Div(myslider, style={"width": "65%", "display": "inline-block"}),
        html.Div(
            solver_options,
            style={"width": "20%", "display": "inline-block", "float": "right"},
        ),
        html.Div(mygraph, style={"width": "70%", "margin": "0 auto"}),
    ]
)


@app.callback(
    [
        Output("stored_stable_fixed_pts", "data"),
        Output("printed_stable_fixed_pts", "children"),
        Output("stored_unstable_fixed_pts", "data"),
        Output("printed_unstable_fixed_pts", "children"),
    ],
    [
        Input("phase_portrait", "clickData"),
        Input("stored_stable_fixed_pts", "data"),
        Input("stored_unstable_fixed_pts", "data"),
    ],
    prevent_initial_call=True,
)
def update_roots(clickData, stable_roots, unstable_roots):
    click_x = clickData["points"][0]["x"]
    x_ = root(diff_eq, click_x).x[0]
    if x_ in set(stable_roots) | set(unstable_roots):
        raise PreventUpdate
    stability = np.sign(derivative(diff_eq, x_, dx=1e-6))
    if stability == 1:
        stable_roots.append(x_)
        return (stable_roots, str(stable_roots), no_update, no_update)
    elif stability == -1:
        unstable_roots.append(x_)
        return (no_update, no_update, unstable_roots, str(unstable_roots))
    else:
        raise PreventUpdate


@app.callback(
    Output("phase_portrait", "figure"),
    [
        Input("stored_stable_fixed_pts", "data"),
        Input("stored_unstable_fixed_pts", "data"),
    ],
    State("phase_portrait", "figure"),
    prevent_initial_call=True,
)
def update_portrait(stable_roots, unstable_roots, fig):
    roots = dict(stable=stable_roots, unstable=unstable_roots)
    ctx = callback_context
    input_source = ctx.triggered[0]["prop_id"].split(".")[0]  # get id of input source
    name = input_source.split("_")[1]  # stable or unstable
    new_portrait = go.Figure(fig)
    new_portrait.update_traces(
        x=roots[name], y=np.zeros(len(roots[name])), selector=dict(name=name)
    )
    return new_portrait


@app.callback(
    Output("graph", "figure"), [Input("myslider", "value"), Input("solver", "value")]
)
def update_graph(x_ini, method):
    x = integrate(t_array=t, diff_eq=diff_eq, y_ini=x_ini * np.pi, method=method)
    dx_dt = diff_eq(x)
    trace_x = go.Scatter(x=t, y=x, mode="lines", name="position")
    trace_dx_dt = go.Scatter(x=t, y=dx_dt, mode="lines", name="speed")
    data = [trace_x, trace_dx_dt]
    layout = go.Layout(title="Evolution over time")
    return {"data": data, "layout": layout}


# Run app
if __name__ == "__main__":
    app.run_server(debug=True, port=3050)
