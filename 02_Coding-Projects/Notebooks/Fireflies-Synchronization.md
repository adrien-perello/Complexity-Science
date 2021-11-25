<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Utility-Scripts" data-toc-modified-id="Utility-Scripts-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Utility Scripts</a></span></li><li><span><a href="#Modeling-rationale" data-toc-modified-id="Modeling-rationale-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Modeling rationale</a></span><ul class="toc-item"><li><span><a href="#The-internal-clock" data-toc-modified-id="The-internal-clock-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>The internal clock</a></span></li><li><span><a href="#Synchronization-process" data-toc-modified-id="Synchronization-process-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Synchronization process</a></span></li></ul></li><li><span><a href="#Cellular-Automata-modeling" data-toc-modified-id="Cellular-Automata-modeling-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Cellular Automata modeling</a></span><ul class="toc-item"><li><span><a href="#Parameters-of-the-simulation" data-toc-modified-id="Parameters-of-the-simulation-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Parameters of the simulation</a></span></li><li><span><a href="#Simulation-script" data-toc-modified-id="Simulation-script-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Simulation script</a></span></li><li><span><a href="#Simulation" data-toc-modified-id="Simulation-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Simulation</a></span></li></ul></li></ul></div>


```python
# before everything, import these libraries (Shift-Enter)
import os
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import ipywidgets as widgets
from fractions import Fraction
from dataclasses import dataclass
```

# Fireflies synchronization


<hr>
&nbsp;


```python
from IPython.lib.display import YouTubeVideo
YouTubeVideo('https://www.youtube.com/watch?v=d77GdblhvEo',width=560,height=315)
```





<iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/https://www.youtube.com/watch?v=d77GdblhvEo"
    frameborder="0"
    allowfullscreen
></iframe>




## Utility Scripts


```python
@dataclass
class Slider:
    """Represent a range of (linear) values as both:
    - an np.array 
    - an ipywidget.Floatslider
    """
    name: str
    start: float
    stop: float
    step: float
    val_ini: float = None

    def __post_init__(self):
        self.val = nparange(self.start, self.stop, self.step)
        if not self.val_ini:
            self.val_ini = np.random.choice(self.val, 1)[0]
        
        self.slider = widgets.FloatSlider(min=self.start,
                                        max=self.stop,
                                        step=self.step,
                                        value=self.val_ini,
                                        description=self.name,
                                        continuous_update=True)


@dataclass
class logSlider:
    """Represent a range of log values as both:
    - an np.array 
    - an ipywidget.FloatLogSlider
    """
    name: str
    start: float
    stop: float
    num: int
    val_ini: float = None
    base: int = 10
    decimals: int = 1

    def __post_init__(self):
        # create numpy array of all values
        self.val = np.around(np.logspace(start=self.start,
                                        stop=self.stop,
                                        num=self.num, 
                                        endpoint=True), self.decimals)
        # check each value is unique
        if self.val.size != np.unique(self.val, return_counts=False).size:
            print(f"WARNING: Repeated values in {self.name}.val"
                ", increase 'decimals' or reduce 'num'")

        # pick initial value if not provided
        if not self.val_ini:
            self.val_ini = np.random.choice(self.val, 1)[0]

        # convert num into step for FloatLogSlider
        step = (self.stop - self.start)/(self.num-1)

        # create slider
        self.slider = widgets.FloatLogSlider(min=self.start,
                                    max=self.stop,
                                    step=step,
                                    value=self.val_ini,
                                    base=self.base,
                                    description=self.name,
                                    readout_format=f'.{self.decimals}f')


@dataclass
class AnimateSlider:
    start: int
    stop: int
    step: float
    name: str = "Press play"
    val_ini: float = None
    interval: float = 100  # time interval (in ms)

    def __post_init__(self):
        # create the play widget
        self.play = widgets.Play(
            min = self.start,
            max = self.stop,
            step = self.step,
            interval = self.interval,
            description = self.name)

        if self.val_ini:
            self.play.value = self.val_ini

        # create a slider for visualization
        self.slider = widgets.IntSlider(
                min = self.start,
                max = self.stop,
                step = self.step)

        # Link the slider and the play widget
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))


def nparange(start, stop, step):
    """Modified np.arange()
        - improve float precision (by use of fractions)
        - includes endpoint

    Args:
        start, stop, step: float (stop is included in array)

    Returns:
        ndarray
    """
    delta, zoom  = get_frac(step)
    
    return np.arange(start * zoom, stop * zoom + delta, delta) / zoom


def get_frac(step, readout_format='.16f', atol=1e-12):
    precision = "{:" + readout_format + "}" 
    frac = Fraction(precision.format(step))
    if frac.denominator > 1/atol:
        print("WARNING: potential Floats inconsistencies due to 'step'"
            " being an irrational number")
    return (frac.numerator, frac.denominator)
```

## Modeling rationale

### The internal clock

- Each firefly has its own individual internal clock (or phase) $\theta$
- $\theta$ has a period T
- $\theta$ varies between 0 and 1 
- Every time the clock reachs 1 (every T times), the firefly flashes 
- After flashing, the clock is reset to 0



$$\begin{align*}
&\theta_{t+1} =  \theta_t + \frac{1}{T} && \text{ (mod 1)} \\
\Rightarrow& \theta_{t+1} - \theta_t = \frac{1}{T} &&  \text{ (mod 1)} \\
\Rightarrow& \frac{\theta_{t+1} - \theta_t}{(t+1) - t} = \frac{1}{T} && \text{ (mod 1)} \\
\Rightarrow& \theta(t) = \frac{t}{T} && \text{ (mod 1)}
\end{align*}$$


```python
# Let's visualize what we just said

# set the variable
t1 = np.linspace(0, 10, 1000)  # time array

# Set a slider to check the influence of the resetting strength
T1 = Slider(name='period T', start=1, stop=10, step=1)

# Set an xarray for all (time, amplitude) value combinations
f1= lambda t, T: np.mod(t/T, 1)
tt1, TT = np.meshgrid(t1, T1.val)
y1 = xr.DataArray(f1(tt1, TT),
            dims=['T', 't'],
            coords={'T': T1.val, 't': t1})


# Set the graph
trace0 = go.Scatter(x=t1, 
                    y=y1.sel(T=T1.val_ini).values)
fig1 = go.FigureWidget([trace0])
fig1.update_layout(template='none',
                    width=800, height=500,
                    title="Flashes of a single firefly",
                    title_x=0.5,
                    xaxis_title="t",
                    yaxis_title='θ')


# Set the callback to update the graph 
def update1(change):
    fig1.data[0].y = y1.sel(T=change['new']).values


# Link the slider and the callback
T1.slider.observe(update1, names='value')

# Display
display(widgets.VBox([T1.slider, fig1]))
```


    VBox(children=(FloatSlider(value=2.0, description='period T', max=10.0, min=1.0, step=1.0), FigureWidget({
       …


### Synchronization process

- When a firefly flashes, it influences its neighbours
- The neighbours slows down or speeds up so as to flash more nearly in
phase on the next cycle
- A simple model satisfying this hypothesis is:


$$\begin{align*}\theta_{t+1} =  \theta_t + \frac{1}{T} + A\sin(\Theta_t - \theta_t)\end{align*}$$

where $\Theta_t$ is the phase of a flashing (neigbhoring) firely[^1].

\cite{pikovsky2001SynchronizationUniversalConcept}


```python
# set the variable
t2 = np.linspace(0,1,1000)  # time array
T2 = 200                    # the period


# Set a slider to check the influence of the resetting strength
A = Slider(name='Amplitude', start=0, stop=1, step=0.01)
A.slider.description_tooltip = "Amplitude of the resetting strength\n "\
                                "(it measures the firefly’s ability to "\
                                "modify its instantaneous frequency)"

# Set an xarray for all (time, amplitude) value combinations
# Option 1: using modulus
f2= lambda t, A: np.mod(t + 1/T2 + A*np.sin(2*np.pi*(1.005-t)), 1)
tt2, AA = np.meshgrid(t2, A.val)
y2 = xr.DataArray(f2(tt2, AA),
            dims=['Amplitude', 't'],
            coords={'Amplitude': A.val, 't': t2})



# Set the graph
trace0 = go.Scatter(x=t2, 
                    y=y2.sel(Amplitude=A.val_ini).values,
                    name='with coupling')
trace1 = go.Scatter(x=t2,
                    y=np.mod(t2+(1/T2),1),
                    name='no coupling')
fig2 = go.FigureWidget([trace0, trace1])
fig2.update_layout(template='none',
                    width=800, height=500,
                    title="Influence of the resetting strength",
                    title_x=0.5,
                    xaxis_title="θ<sub>t<sub>",
                    yaxis_title="θ<sub>t+1<sub>",
                    legend_title='Click to deselect', 
                    legend_title_font=dict(size=16),
                    legend_title_font_color='FireBrick')


# Set the callback to update the graph 
def update2(change):
    with fig2.batch_update():
        fig2.data[0].y = y2.sel(Amplitude=change['new']).values

# Link the slider and the callback
A.slider.observe(update2, names='value')


# Display
display(widgets.VBox([A.slider, fig2]))
```


    VBox(children=(FloatSlider(value=0.08, description='Amplitude', description_tooltip='Amplitude of the resettin…


## Cellular Automata modeling

### Parameters of the simulation


```python
@dataclass
class Parameters:
    """This class contains the parameters of the simulation.
    """

    # Size of the grid
    M: int = 10            # number of cells in the x direction
    N: int = 10            # number of cells in the y direction
    fireflies_density: float = 1  # how much of the grid is populated

    # Coupling parameters
    coupling_value: float = 0.1    # float (between 0 and +/- 0.3)
    neighbor_distance: float = 3  # Size of neighbourhood radius

    # Simulation settings
    nb_periods: int = 20
    time_step_per_period: int = 100  # dt = 1/time_step_per_period
```

### Simulation script


```python
def fireflies_simulator(par):
    """This function simulates the fireflies system, computes the order
    parameter and shows figures of the system state during the simulation.

    Args:
        - Parameters

    Returns:
        The value of the order parameter over time
    """
    time_steps = np.arange(start=0,
                            stop=par.nb_periods * par.time_step_per_period,
                            step=1)
    phases = np.zeros((time_steps.size+1, par.N, par.M))  # 1 grid (MxN) per time step
    phases[0] = np.random.random((par.N, par.M))          # random initial state

    # empty some cells to ensure the right fireflies density
    nb_empty_cell = np.around((1-par.fireflies_density)*par.N*par.M, 0).astype(int)
    ind = np.random.choice(phases[0].size, size=nb_empty_cell, replace=False)
    phases[0].ravel()[ind] = np.nan

    neighbors = compute_neighborhood(phases[0], par.N, par.M, par.neighbor_distance)
    phase_increment = 1 / par.time_step_per_period

    for t in time_steps:
        phases[t+1] = phases[t] + phase_increment
        glow_idx = np.array(np.nonzero(phases[t+1]>1)).T
        ids = [np.ravel_multi_index(tup, (par.N, par.M)) for tup in glow_idx]
        for ind in ids:
            i,j = np.unravel_index(ind, (par.N, par.M))
            phases[t+1][neighbors[ind]] = nudge(phases[t+1][neighbors[ind]], 
                                                phases[t+1][i,j],
                                                par.coupling_value)

    return  phases


def compute_neighborhood(grid, N, M, r):
    """For every fireflies, compute a mask array of neighboring fireflies.

    Args:
        grid (ndarray): phase state of each firefly
        N (int): number of cells in the x direction
        M (int): number of cells in the y direction
        r (float): radius of neighbour_distance

    Returns:
        dict: keys: index (ravel) of each firefly
                values: mask array with neighbour fireflies
    """
    neighbors = dict.fromkeys((range(N*M)), [])
    occupied_cells = ~np.isnan(grid)
    for i in range(N):
        for j in range(M):
            if np.isnan(grid[i,j]):
                continue
            y, x = np.ogrid[-i:N-i, -j:M-j]
            mask = (x**2 + y**2 <= r**2)*occupied_cells  # only keep occupied neighboring cells
            # mask[i,j] = False                            # don't include the cell istelf
            neighbors[np.ravel_multi_index((i,j), (N,M))] = mask
    return neighbors


def nudge(neighbor_phases, flash_phase, amplitude):
    """Nudge the neighboring fireflies.

    Args:
        neighbor_phases (ndarray): phases of all the fireflies to nudge
        flash_phase (float): phase of the flashing firefly
        amplitude (float): resetting strength / coupling value
    """
    phase_diff = flash_phase - neighbor_phases
    res = neighbor_phases + amplitude*np.sin(2*np.pi*(phase_diff))
    return np.mod(res, 1)
```

### Simulation


```python
# Let's initiate the (default) parameters for a simulation
settings = Parameters(N=20, M=20, 
                    fireflies_density=0.8, 
                    coupling_value=0.01, 
                    neighbor_distance= 2,
                    time_step_per_period=100, 
                    nb_periods=100
)

# And let's run the simulation
heatmaps = fireflies_simulator(settings)
```


```python
# Let's visualiza the simulation

mymin = 0
mymax = heatmaps.shape[0]-1
mystep = 10


# Set the figure
fig5 = go.FigureWidget(
    data=go.Heatmap(z=heatmaps[0], colorscale='Hot', reversescale=True),
    layout=go.Layout(title="Fireflies Simulator"))
fig5.data[0].update(zmin=0, zmax=1)  # fix the colorscale to [0,1]


# Set the callback to update the graph 
def update5(change):
    fig5.data[0].z = heatmaps[change['new']]


# Create the animation display and link it to the callback function
myanimation = AnimateSlider(start=mymin, stop=mymax, step=mystep)
myanimation.slider.observe(update5, names='value')
controllers = widgets.HBox([myanimation.play, myanimation.slider])


# Display
display(widgets.VBox([controllers, fig5]))
```


    VBox(children=(HBox(children=(Play(value=0, description='Press play', max=10000, step=10), IntSlider(value=0, …



```python
# Statistical results

t3 = np.arange(heatmaps.shape[0])
avg = np.nanmean(heatmaps, axis=(1,2))
std = np.nanstd(heatmaps, axis=(1,2))
```


```python
#show light intensity over time (the last 500 time steps)

trace0 = go.Scatter(x=t3[-1000:], y=avg[-1000:], name='average')
trace1 = go.Scatter(x=t3[-1000:], y=std[-1000:], name='std dev')

fig3 = go.FigureWidget([trace0, trace1])
fig3.update_yaxes(range=[0, 1])       # fix the scale to [0,1] for easy comparison
fig3.update_layout(template='none',
                    width=800, height=500,
                    title="Light intensity",
                    title_x=0.5,
                    xaxis_title="t",
                    yaxis_title="I")
fig3
```


    FigureWidget({
        'data': [{'name': 'average',
                  'type': 'scatter',
                  'uid': 'f855cbb…


<!--bibtex

@Article{PER-GRA:2007,
  Author    = {P\'erez, Fernando and Granger, Brian E.},
  Title     = {{IP}ython: a System for Interactive Scientific Computing},
  Journal   = {Computing in Science and Engineering},
  Volume    = {9},
  Number    = {3},
  Pages     = {21--29},
  month     = may,
  year      = 2007,
  url       = "http://ipython.org",
  ISSN      = "1521-9615",
  doi       = {10.1109/MCSE.2007.53},
  publisher = {IEEE Computer Society},
}

@article{Papa2007,
  author = {Papa, David A. and Markov, Igor L.},
  journal = {Approximation algorithms and metaheuristics},
  pages = {1--38},
  title = {{Hypergraph partitioning and clustering}},
  url = {http://www.podload.org/pubs/book/part\_survey.pdf},
  year = {2007}
}

-->

Examples of citations: [CITE](#cite-PER-GRA:2007) or [CITE](#cite-Papa2007).

<!--


```python
#automatic document conversion to markdown and then to word
#first convert the ipython notebook paper.ipynb to markdown
os.system("jupyter nbconvert --to markdown Fireflies-Synchronization.ipynb")
    
 #next convert markdown to ms word
conversion = f"pandoc -s Fireflies-Synchronization.md --citeproc --bibliography Fireflies-Synchronization.bib --csl=apa.csl"

os.system(conversion)
```

    [NbConvertApp] Converting notebook Fireflies-Synchronization.ipynb to markdown
    [NbConvertApp] Writing 92908 bytes to Fireflies-Synchronization.md


    <!DOCTYPE html>
    <html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">
    <head>
      <meta charset="utf-8" />
      <meta name="generator" content="pandoc" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
      <title>Fireflies-Synchronization</title>
      <style>
        html {
          line-height: 1.5;
          font-family: Georgia, serif;
          font-size: 20px;
          color: #1a1a1a;
          background-color: #fdfdfd;
        }
        body {
          margin: 0 auto;
          max-width: 36em;
          padding-left: 50px;
          padding-right: 50px;
          padding-top: 50px;
          padding-bottom: 50px;
          hyphens: auto;
          overflow-wrap: break-word;
          text-rendering: optimizeLegibility;
          font-kerning: normal;
        }
        @media (max-width: 600px) {
          body {
            font-size: 0.9em;
            padding: 1em;
          }
        }
        @media print {
          body {
            background-color: transparent;
            color: black;
            font-size: 12pt;
          }
          p, h2, h3 {
            orphans: 3;
            widows: 3;
          }
          h2, h3, h4 {
            page-break-after: avoid;
          }
        }
        p {
          margin: 1em 0;
        }
        a {
          color: #1a1a1a;
        }
        a:visited {
          color: #1a1a1a;
        }
        img {
          max-width: 100%;
        }
        h1, h2, h3, h4, h5, h6 {
          margin-top: 1.4em;
        }
        h5, h6 {
          font-size: 1em;
          font-style: italic;
        }
        h6 {
          font-weight: normal;
        }
        ol, ul {
          padding-left: 1.7em;
          margin-top: 1em;
        }
        li > ol, li > ul {
          margin-top: 0;
        }
        blockquote {
          margin: 1em 0 1em 1.7em;
          padding-left: 1em;
          border-left: 2px solid #e6e6e6;
          color: #606060;
        }
        code {
          font-family: Menlo, Monaco, 'Lucida Console', Consolas, monospace;
          font-size: 85%;
          margin: 0;
        }
        pre {
          margin: 1em 0;
          overflow: auto;
        }
        pre code {
          padding: 0;
          overflow: visible;
          overflow-wrap: normal;
        }
        .sourceCode {
         background-color: transparent;
         overflow: visible;
        }
        hr {
          background-color: #1a1a1a;
          border: none;
          height: 1px;
          margin: 1em 0;
        }
        table {
          margin: 1em 0;
          border-collapse: collapse;
          width: 100%;
          overflow-x: auto;
          display: block;
          font-variant-numeric: lining-nums tabular-nums;
        }
        table caption {
          margin-bottom: 0.75em;
        }
        tbody {
          margin-top: 0.5em;
          border-top: 1px solid #1a1a1a;
          border-bottom: 1px solid #1a1a1a;
        }
        th {
          border-top: 1px solid #1a1a1a;
          padding: 0.25em 0.5em 0.25em 0.5em;
        }
        td {
          padding: 0.125em 0.5em 0.25em 0.5em;
        }
        header {
          margin-bottom: 4em;
          text-align: center;
        }
        #TOC li {
          list-style: none;
        }
        #TOC a:not(:hover) {
          text-decoration: none;
        }
        code{white-space: pre-wrap;}
        span.smallcaps{font-variant: small-caps;}
        span.underline{text-decoration: underline;}
        div.column{display: inline-block; vertical-align: top; width: 50%;}
        div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;}
        ul.task-list{list-style: none;}
        pre > code.sourceCode { white-space: pre; position: relative; }
        pre > code.sourceCode > span { display: inline-block; line-height: 1.25; }
        pre > code.sourceCode > span:empty { height: 1.2em; }
        .sourceCode { overflow: visible; }
        code.sourceCode > span { color: inherit; text-decoration: inherit; }
        div.sourceCode { margin: 1em 0; }
        pre.sourceCode { margin: 0; }
        @media screen {
        div.sourceCode { overflow: auto; }
        }
        @media print {
        pre > code.sourceCode { white-space: pre-wrap; }
        pre > code.sourceCode > span { text-indent: -5em; padding-left: 5em; }
        }
        pre.numberSource code
          { counter-reset: source-line 0; }
        pre.numberSource code > span
          { position: relative; left: -4em; counter-increment: source-line; }
        pre.numberSource code > span > a:first-child::before
          { content: counter(source-line);
            position: relative; left: -1em; text-align: right; vertical-align: baseline;
            border: none; display: inline-block;
            -webkit-touch-callout: none; -webkit-user-select: none;
            -khtml-user-select: none; -moz-user-select: none;
            -ms-user-select: none; user-select: none;
            padding: 0 4px; width: 4em;
            color: #aaaaaa;
          }
        pre.numberSource { margin-left: 3em; border-left: 1px solid #aaaaaa;  padding-left: 4px; }
        div.sourceCode
          {   }
        @media screen {
        pre > code.sourceCode > span > a:first-child::before { text-decoration: underline; }
        }
        code span.al { color: #ff0000; font-weight: bold; } /* Alert */
        code span.an { color: #60a0b0; font-weight: bold; font-style: italic; } /* Annotation */
        code span.at { color: #7d9029; } /* Attribute */
        code span.bn { color: #40a070; } /* BaseN */
        code span.bu { } /* BuiltIn */
        code span.cf { color: #007020; font-weight: bold; } /* ControlFlow */
        code span.ch { color: #4070a0; } /* Char */
        code span.cn { color: #880000; } /* Constant */
        code span.co { color: #60a0b0; font-style: italic; } /* Comment */
        code span.cv { color: #60a0b0; font-weight: bold; font-style: italic; } /* CommentVar */
        code span.do { color: #ba2121; font-style: italic; } /* Documentation */
        code span.dt { color: #902000; } /* DataType */
        code span.dv { color: #40a070; } /* DecVal */
        code span.er { color: #ff0000; font-weight: bold; } /* Error */
        code span.ex { } /* Extension */
        code span.fl { color: #40a070; } /* Float */
        code span.fu { color: #06287e; } /* Function */
        code span.im { } /* Import */
        code span.in { color: #60a0b0; font-weight: bold; font-style: italic; } /* Information */
        code span.kw { color: #007020; font-weight: bold; } /* Keyword */
        code span.op { color: #666666; } /* Operator */
        code span.ot { color: #007020; } /* Other */
        code span.pp { color: #bc7a00; } /* Preprocessor */
        code span.sc { color: #4070a0; } /* SpecialChar */
        code span.ss { color: #bb6688; } /* SpecialString */
        code span.st { color: #4070a0; } /* String */
        code span.va { color: #19177c; } /* Variable */
        code span.vs { color: #4070a0; } /* VerbatimString */
        code span.wa { color: #60a0b0; font-weight: bold; font-style: italic; } /* Warning */
        .display.math{display: block; text-align: center; margin: 0.5rem auto;}
      </style>
      <!--[if lt IE 9]>
        <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
      <![endif]-->
    </head>
    <body>
    <h1>
    Table of Contents<span class="tocSkip"></span>
    </h1>
    <div class="toc">
    <ul class="toc-item">
    <li>
    <span><a href="#Utility-Scripts" data-toc-modified-id="Utility-Scripts-1"><span class="toc-item-num">1  </span>Utility Scripts</a></span>
    </li>
    <li>
    <span><a href="#Modeling-rationale" data-toc-modified-id="Modeling-rationale-2"><span class="toc-item-num">2  </span>Modeling rationale</a></span>
    <ul class="toc-item">
    <li>
    <span><a href="#The-internal-clock" data-toc-modified-id="The-internal-clock-2.1"><span class="toc-item-num">2.1  </span>The internal clock</a></span>
    </li>
    <li>
    <span><a href="#Synchronization-process" data-toc-modified-id="Synchronization-process-2.2"><span class="toc-item-num">2.2  </span>Synchronization process</a></span>
    </li>
    </ul>
    </li>
    <li>
    <span><a href="#Cellular-Automata-modeling" data-toc-modified-id="Cellular-Automata-modeling-3"><span class="toc-item-num">3  </span>Cellular Automata modeling</a></span>
    <ul class="toc-item">
    <li>
    <span><a href="#Parameters-of-the-simulation" data-toc-modified-id="Parameters-of-the-simulation-3.1"><span class="toc-item-num">3.1  </span>Parameters of the simulation</a></span>
    </li>
    <li>
    <span><a href="#Simulation-script" data-toc-modified-id="Simulation-script-3.2"><span class="toc-item-num">3.2  </span>Simulation script</a></span>
    </li>
    <li>
    <span><a href="#Simulation" data-toc-modified-id="Simulation-3.3"><span class="toc-item-num">3.3  </span>Simulation</a></span>
    </li>
    </ul>
    </li>
    </ul>
    </div>
    <div class="sourceCode" id="cb1"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="co"># before everything, import these libraries (Shift-Enter)</span></span>
    <span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> os</span>
    <span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> numpy <span class="im">as</span> np</span>
    <span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> xarray <span class="im">as</span> xr</span>
    <span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> plotly.graph_objects <span class="im">as</span> go</span>
    <span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> ipywidgets <span class="im">as</span> widgets</span>
    <span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> fractions <span class="im">import</span> Fraction</span>
    <span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> dataclasses <span class="im">import</span> dataclass</span></code></pre></div>
    <h1 id="fireflies-synchronization">Fireflies synchronization</h1>
    <hr>
    <p> </p>
    <div class="sourceCode" id="cb2"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> IPython.lib.display <span class="im">import</span> YouTubeVideo</span>
    <span id="cb2-2"><a href="#cb2-2" aria-hidden="true" tabindex="-1"></a>YouTubeVideo(<span class="st">&#39;https://www.youtube.com/watch?v=d77GdblhvEo&#39;</span>,width<span class="op">=</span><span class="dv">560</span>,height<span class="op">=</span><span class="dv">315</span>)</span></code></pre></div>
    <iframe width="560" height="315" src="https://www.youtube.com/embed/https://www.youtube.com/watch?v=d77GdblhvEo" frameborder="0" allowfullscreen>
    </iframe>
    <h2 id="utility-scripts">Utility Scripts</h2>
    <div class="sourceCode" id="cb3"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a><span class="at">@dataclass</span></span>
    <span id="cb3-2"><a href="#cb3-2" aria-hidden="true" tabindex="-1"></a><span class="kw">class</span> Slider:</span>
    <span id="cb3-3"><a href="#cb3-3" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;Represent a range of (linear) values as both:</span></span>
    <span id="cb3-4"><a href="#cb3-4" aria-hidden="true" tabindex="-1"></a><span class="co">    - an np.array </span></span>
    <span id="cb3-5"><a href="#cb3-5" aria-hidden="true" tabindex="-1"></a><span class="co">    - an ipywidget.Floatslider</span></span>
    <span id="cb3-6"><a href="#cb3-6" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb3-7"><a href="#cb3-7" aria-hidden="true" tabindex="-1"></a>    name: <span class="bu">str</span></span>
    <span id="cb3-8"><a href="#cb3-8" aria-hidden="true" tabindex="-1"></a>    start: <span class="bu">float</span></span>
    <span id="cb3-9"><a href="#cb3-9" aria-hidden="true" tabindex="-1"></a>    stop: <span class="bu">float</span></span>
    <span id="cb3-10"><a href="#cb3-10" aria-hidden="true" tabindex="-1"></a>    step: <span class="bu">float</span></span>
    <span id="cb3-11"><a href="#cb3-11" aria-hidden="true" tabindex="-1"></a>    val_ini: <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span></span>
    <span id="cb3-12"><a href="#cb3-12" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-13"><a href="#cb3-13" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> __post_init__(<span class="va">self</span>):</span>
    <span id="cb3-14"><a href="#cb3-14" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.val <span class="op">=</span> nparange(<span class="va">self</span>.start, <span class="va">self</span>.stop, <span class="va">self</span>.step)</span>
    <span id="cb3-15"><a href="#cb3-15" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> <span class="kw">not</span> <span class="va">self</span>.val_ini:</span>
    <span id="cb3-16"><a href="#cb3-16" aria-hidden="true" tabindex="-1"></a>            <span class="va">self</span>.val_ini <span class="op">=</span> np.random.choice(<span class="va">self</span>.val, <span class="dv">1</span>)[<span class="dv">0</span>]</span>
    <span id="cb3-17"><a href="#cb3-17" aria-hidden="true" tabindex="-1"></a>        </span>
    <span id="cb3-18"><a href="#cb3-18" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.slider <span class="op">=</span> widgets.FloatSlider(<span class="bu">min</span><span class="op">=</span><span class="va">self</span>.start,</span>
    <span id="cb3-19"><a href="#cb3-19" aria-hidden="true" tabindex="-1"></a>                                        <span class="bu">max</span><span class="op">=</span><span class="va">self</span>.stop,</span>
    <span id="cb3-20"><a href="#cb3-20" aria-hidden="true" tabindex="-1"></a>                                        step<span class="op">=</span><span class="va">self</span>.step,</span>
    <span id="cb3-21"><a href="#cb3-21" aria-hidden="true" tabindex="-1"></a>                                        value<span class="op">=</span><span class="va">self</span>.val_ini,</span>
    <span id="cb3-22"><a href="#cb3-22" aria-hidden="true" tabindex="-1"></a>                                        description<span class="op">=</span><span class="va">self</span>.name,</span>
    <span id="cb3-23"><a href="#cb3-23" aria-hidden="true" tabindex="-1"></a>                                        continuous_update<span class="op">=</span><span class="va">True</span>)</span>
    <span id="cb3-24"><a href="#cb3-24" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-25"><a href="#cb3-25" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-26"><a href="#cb3-26" aria-hidden="true" tabindex="-1"></a><span class="at">@dataclass</span></span>
    <span id="cb3-27"><a href="#cb3-27" aria-hidden="true" tabindex="-1"></a><span class="kw">class</span> logSlider:</span>
    <span id="cb3-28"><a href="#cb3-28" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;Represent a range of log values as both:</span></span>
    <span id="cb3-29"><a href="#cb3-29" aria-hidden="true" tabindex="-1"></a><span class="co">    - an np.array </span></span>
    <span id="cb3-30"><a href="#cb3-30" aria-hidden="true" tabindex="-1"></a><span class="co">    - an ipywidget.FloatLogSlider</span></span>
    <span id="cb3-31"><a href="#cb3-31" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb3-32"><a href="#cb3-32" aria-hidden="true" tabindex="-1"></a>    name: <span class="bu">str</span></span>
    <span id="cb3-33"><a href="#cb3-33" aria-hidden="true" tabindex="-1"></a>    start: <span class="bu">float</span></span>
    <span id="cb3-34"><a href="#cb3-34" aria-hidden="true" tabindex="-1"></a>    stop: <span class="bu">float</span></span>
    <span id="cb3-35"><a href="#cb3-35" aria-hidden="true" tabindex="-1"></a>    num: <span class="bu">int</span></span>
    <span id="cb3-36"><a href="#cb3-36" aria-hidden="true" tabindex="-1"></a>    val_ini: <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span></span>
    <span id="cb3-37"><a href="#cb3-37" aria-hidden="true" tabindex="-1"></a>    base: <span class="bu">int</span> <span class="op">=</span> <span class="dv">10</span></span>
    <span id="cb3-38"><a href="#cb3-38" aria-hidden="true" tabindex="-1"></a>    decimals: <span class="bu">int</span> <span class="op">=</span> <span class="dv">1</span></span>
    <span id="cb3-39"><a href="#cb3-39" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-40"><a href="#cb3-40" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> __post_init__(<span class="va">self</span>):</span>
    <span id="cb3-41"><a href="#cb3-41" aria-hidden="true" tabindex="-1"></a>        <span class="co"># create numpy array of all values</span></span>
    <span id="cb3-42"><a href="#cb3-42" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.val <span class="op">=</span> np.around(np.logspace(start<span class="op">=</span><span class="va">self</span>.start,</span>
    <span id="cb3-43"><a href="#cb3-43" aria-hidden="true" tabindex="-1"></a>                                        stop<span class="op">=</span><span class="va">self</span>.stop,</span>
    <span id="cb3-44"><a href="#cb3-44" aria-hidden="true" tabindex="-1"></a>                                        num<span class="op">=</span><span class="va">self</span>.num, </span>
    <span id="cb3-45"><a href="#cb3-45" aria-hidden="true" tabindex="-1"></a>                                        endpoint<span class="op">=</span><span class="va">True</span>), <span class="va">self</span>.decimals)</span>
    <span id="cb3-46"><a href="#cb3-46" aria-hidden="true" tabindex="-1"></a>        <span class="co"># check each value is unique</span></span>
    <span id="cb3-47"><a href="#cb3-47" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> <span class="va">self</span>.val.size <span class="op">!=</span> np.unique(<span class="va">self</span>.val, return_counts<span class="op">=</span><span class="va">False</span>).size:</span>
    <span id="cb3-48"><a href="#cb3-48" aria-hidden="true" tabindex="-1"></a>            <span class="bu">print</span>(<span class="ss">f&quot;WARNING: Repeated values in </span><span class="sc">{</span><span class="va">self</span><span class="sc">.</span>name<span class="sc">}</span><span class="ss">.val&quot;</span></span>
    <span id="cb3-49"><a href="#cb3-49" aria-hidden="true" tabindex="-1"></a>                <span class="st">&quot;, increase &#39;decimals&#39; or reduce &#39;num&#39;&quot;</span>)</span>
    <span id="cb3-50"><a href="#cb3-50" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-51"><a href="#cb3-51" aria-hidden="true" tabindex="-1"></a>        <span class="co"># pick initial value if not provided</span></span>
    <span id="cb3-52"><a href="#cb3-52" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> <span class="kw">not</span> <span class="va">self</span>.val_ini:</span>
    <span id="cb3-53"><a href="#cb3-53" aria-hidden="true" tabindex="-1"></a>            <span class="va">self</span>.val_ini <span class="op">=</span> np.random.choice(<span class="va">self</span>.val, <span class="dv">1</span>)[<span class="dv">0</span>]</span>
    <span id="cb3-54"><a href="#cb3-54" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-55"><a href="#cb3-55" aria-hidden="true" tabindex="-1"></a>        <span class="co"># convert num into step for FloatLogSlider</span></span>
    <span id="cb3-56"><a href="#cb3-56" aria-hidden="true" tabindex="-1"></a>        step <span class="op">=</span> (<span class="va">self</span>.stop <span class="op">-</span> <span class="va">self</span>.start)<span class="op">/</span>(<span class="va">self</span>.num<span class="op">-</span><span class="dv">1</span>)</span>
    <span id="cb3-57"><a href="#cb3-57" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-58"><a href="#cb3-58" aria-hidden="true" tabindex="-1"></a>        <span class="co"># create slider</span></span>
    <span id="cb3-59"><a href="#cb3-59" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.slider <span class="op">=</span> widgets.FloatLogSlider(<span class="bu">min</span><span class="op">=</span><span class="va">self</span>.start,</span>
    <span id="cb3-60"><a href="#cb3-60" aria-hidden="true" tabindex="-1"></a>                                    <span class="bu">max</span><span class="op">=</span><span class="va">self</span>.stop,</span>
    <span id="cb3-61"><a href="#cb3-61" aria-hidden="true" tabindex="-1"></a>                                    step<span class="op">=</span>step,</span>
    <span id="cb3-62"><a href="#cb3-62" aria-hidden="true" tabindex="-1"></a>                                    value<span class="op">=</span><span class="va">self</span>.val_ini,</span>
    <span id="cb3-63"><a href="#cb3-63" aria-hidden="true" tabindex="-1"></a>                                    base<span class="op">=</span><span class="va">self</span>.base,</span>
    <span id="cb3-64"><a href="#cb3-64" aria-hidden="true" tabindex="-1"></a>                                    description<span class="op">=</span><span class="va">self</span>.name,</span>
    <span id="cb3-65"><a href="#cb3-65" aria-hidden="true" tabindex="-1"></a>                                    readout_format<span class="op">=</span><span class="ss">f&#39;.</span><span class="sc">{</span><span class="va">self</span><span class="sc">.</span>decimals<span class="sc">}</span><span class="ss">f&#39;</span>)</span>
    <span id="cb3-66"><a href="#cb3-66" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-67"><a href="#cb3-67" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-68"><a href="#cb3-68" aria-hidden="true" tabindex="-1"></a><span class="at">@dataclass</span></span>
    <span id="cb3-69"><a href="#cb3-69" aria-hidden="true" tabindex="-1"></a><span class="kw">class</span> AnimateSlider:</span>
    <span id="cb3-70"><a href="#cb3-70" aria-hidden="true" tabindex="-1"></a>    start: <span class="bu">int</span></span>
    <span id="cb3-71"><a href="#cb3-71" aria-hidden="true" tabindex="-1"></a>    stop: <span class="bu">int</span></span>
    <span id="cb3-72"><a href="#cb3-72" aria-hidden="true" tabindex="-1"></a>    step: <span class="bu">float</span></span>
    <span id="cb3-73"><a href="#cb3-73" aria-hidden="true" tabindex="-1"></a>    name: <span class="bu">str</span> <span class="op">=</span> <span class="st">&quot;Press play&quot;</span></span>
    <span id="cb3-74"><a href="#cb3-74" aria-hidden="true" tabindex="-1"></a>    val_ini: <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span></span>
    <span id="cb3-75"><a href="#cb3-75" aria-hidden="true" tabindex="-1"></a>    interval: <span class="bu">float</span> <span class="op">=</span> <span class="dv">100</span>  <span class="co"># time interval (in ms)</span></span>
    <span id="cb3-76"><a href="#cb3-76" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-77"><a href="#cb3-77" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> __post_init__(<span class="va">self</span>):</span>
    <span id="cb3-78"><a href="#cb3-78" aria-hidden="true" tabindex="-1"></a>        <span class="co"># create the play widget</span></span>
    <span id="cb3-79"><a href="#cb3-79" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.play <span class="op">=</span> widgets.Play(</span>
    <span id="cb3-80"><a href="#cb3-80" aria-hidden="true" tabindex="-1"></a>            <span class="bu">min</span> <span class="op">=</span> <span class="va">self</span>.start,</span>
    <span id="cb3-81"><a href="#cb3-81" aria-hidden="true" tabindex="-1"></a>            <span class="bu">max</span> <span class="op">=</span> <span class="va">self</span>.stop,</span>
    <span id="cb3-82"><a href="#cb3-82" aria-hidden="true" tabindex="-1"></a>            step <span class="op">=</span> <span class="va">self</span>.step,</span>
    <span id="cb3-83"><a href="#cb3-83" aria-hidden="true" tabindex="-1"></a>            interval <span class="op">=</span> <span class="va">self</span>.interval,</span>
    <span id="cb3-84"><a href="#cb3-84" aria-hidden="true" tabindex="-1"></a>            description <span class="op">=</span> <span class="va">self</span>.name)</span>
    <span id="cb3-85"><a href="#cb3-85" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-86"><a href="#cb3-86" aria-hidden="true" tabindex="-1"></a>        <span class="cf">if</span> <span class="va">self</span>.val_ini:</span>
    <span id="cb3-87"><a href="#cb3-87" aria-hidden="true" tabindex="-1"></a>            <span class="va">self</span>.play.value <span class="op">=</span> <span class="va">self</span>.val_ini</span>
    <span id="cb3-88"><a href="#cb3-88" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-89"><a href="#cb3-89" aria-hidden="true" tabindex="-1"></a>        <span class="co"># create a slider for visualization</span></span>
    <span id="cb3-90"><a href="#cb3-90" aria-hidden="true" tabindex="-1"></a>        <span class="va">self</span>.slider <span class="op">=</span> widgets.IntSlider(</span>
    <span id="cb3-91"><a href="#cb3-91" aria-hidden="true" tabindex="-1"></a>                <span class="bu">min</span> <span class="op">=</span> <span class="va">self</span>.start,</span>
    <span id="cb3-92"><a href="#cb3-92" aria-hidden="true" tabindex="-1"></a>                <span class="bu">max</span> <span class="op">=</span> <span class="va">self</span>.stop,</span>
    <span id="cb3-93"><a href="#cb3-93" aria-hidden="true" tabindex="-1"></a>                step <span class="op">=</span> <span class="va">self</span>.step)</span>
    <span id="cb3-94"><a href="#cb3-94" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-95"><a href="#cb3-95" aria-hidden="true" tabindex="-1"></a>        <span class="co"># Link the slider and the play widget</span></span>
    <span id="cb3-96"><a href="#cb3-96" aria-hidden="true" tabindex="-1"></a>        widgets.jslink((<span class="va">self</span>.play, <span class="st">&#39;value&#39;</span>), (<span class="va">self</span>.slider, <span class="st">&#39;value&#39;</span>))</span>
    <span id="cb3-97"><a href="#cb3-97" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-98"><a href="#cb3-98" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-99"><a href="#cb3-99" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> nparange(start, stop, step):</span>
    <span id="cb3-100"><a href="#cb3-100" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;Modified np.arange()</span></span>
    <span id="cb3-101"><a href="#cb3-101" aria-hidden="true" tabindex="-1"></a><span class="co">        - improve float precision (by use of fractions)</span></span>
    <span id="cb3-102"><a href="#cb3-102" aria-hidden="true" tabindex="-1"></a><span class="co">        - includes endpoint</span></span>
    <span id="cb3-103"><a href="#cb3-103" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-104"><a href="#cb3-104" aria-hidden="true" tabindex="-1"></a><span class="co">    Args:</span></span>
    <span id="cb3-105"><a href="#cb3-105" aria-hidden="true" tabindex="-1"></a><span class="co">        start, stop, step: float (stop is included in array)</span></span>
    <span id="cb3-106"><a href="#cb3-106" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-107"><a href="#cb3-107" aria-hidden="true" tabindex="-1"></a><span class="co">    Returns:</span></span>
    <span id="cb3-108"><a href="#cb3-108" aria-hidden="true" tabindex="-1"></a><span class="co">        ndarray</span></span>
    <span id="cb3-109"><a href="#cb3-109" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb3-110"><a href="#cb3-110" aria-hidden="true" tabindex="-1"></a>    delta, zoom  <span class="op">=</span> get_frac(step)</span>
    <span id="cb3-111"><a href="#cb3-111" aria-hidden="true" tabindex="-1"></a>    </span>
    <span id="cb3-112"><a href="#cb3-112" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> np.arange(start <span class="op">*</span> zoom, stop <span class="op">*</span> zoom <span class="op">+</span> delta, delta) <span class="op">/</span> zoom</span>
    <span id="cb3-113"><a href="#cb3-113" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-114"><a href="#cb3-114" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb3-115"><a href="#cb3-115" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> get_frac(step, readout_format<span class="op">=</span><span class="st">&#39;.16f&#39;</span>, atol<span class="op">=</span><span class="fl">1e-12</span>):</span>
    <span id="cb3-116"><a href="#cb3-116" aria-hidden="true" tabindex="-1"></a>    precision <span class="op">=</span> <span class="st">&quot;{:&quot;</span> <span class="op">+</span> readout_format <span class="op">+</span> <span class="st">&quot;}&quot;</span> </span>
    <span id="cb3-117"><a href="#cb3-117" aria-hidden="true" tabindex="-1"></a>    frac <span class="op">=</span> Fraction(precision.<span class="bu">format</span>(step))</span>
    <span id="cb3-118"><a href="#cb3-118" aria-hidden="true" tabindex="-1"></a>    <span class="cf">if</span> frac.denominator <span class="op">&gt;</span> <span class="dv">1</span><span class="op">/</span>atol:</span>
    <span id="cb3-119"><a href="#cb3-119" aria-hidden="true" tabindex="-1"></a>        <span class="bu">print</span>(<span class="st">&quot;WARNING: potential Floats inconsistencies due to &#39;step&#39;&quot;</span></span>
    <span id="cb3-120"><a href="#cb3-120" aria-hidden="true" tabindex="-1"></a>            <span class="st">&quot; being an irrational number&quot;</span>)</span>
    <span id="cb3-121"><a href="#cb3-121" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> (frac.numerator, frac.denominator)</span></code></pre></div>
    <h2 id="modeling-rationale">Modeling rationale</h2>
    <h3 id="the-internal-clock">The internal clock</h3>
    <ul>
    <li>Each firefly has its own individual internal clock (or phase) <span class="math inline"><em>θ</em></span></li>
    <li><span class="math inline"><em>θ</em></span> has a period T</li>
    <li><span class="math inline"><em>θ</em></span> varies between 0 and 1</li>
    <li>Every time the clock reachs 1 (every T times), the firefly flashes</li>
    <li>After flashing, the clock is reset to 0</li>
    </ul>
    <p><span class="math display">$$\begin{align*}
    &amp;\theta_{t+1} =  \theta_t + \frac{1}{T} &amp;&amp; \text{ (mod 1)} \\
    \Rightarrow&amp; \theta_{t+1} - \theta_t = \frac{1}{T} &amp;&amp;  \text{ (mod 1)} \\
    \Rightarrow&amp; \frac{\theta_{t+1} - \theta_t}{(t+1) - t} = \frac{1}{T} &amp;&amp; \text{ (mod 1)} \\
    \Rightarrow&amp; \theta(t) = \frac{t}{T} &amp;&amp; \text{ (mod 1)}
    \end{align*}$$</span></p>
    <div class="sourceCode" id="cb4"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Let&#39;s visualize what we just said</span></span>
    <span id="cb4-2"><a href="#cb4-2" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-3"><a href="#cb4-3" aria-hidden="true" tabindex="-1"></a><span class="co"># set the variable</span></span>
    <span id="cb4-4"><a href="#cb4-4" aria-hidden="true" tabindex="-1"></a>t1 <span class="op">=</span> np.linspace(<span class="dv">0</span>, <span class="dv">10</span>, <span class="dv">1000</span>)  <span class="co"># time array</span></span>
    <span id="cb4-5"><a href="#cb4-5" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-6"><a href="#cb4-6" aria-hidden="true" tabindex="-1"></a><span class="co"># Set a slider to check the influence of the resetting strength</span></span>
    <span id="cb4-7"><a href="#cb4-7" aria-hidden="true" tabindex="-1"></a>T1 <span class="op">=</span> Slider(name<span class="op">=</span><span class="st">&#39;period T&#39;</span>, start<span class="op">=</span><span class="dv">1</span>, stop<span class="op">=</span><span class="dv">10</span>, step<span class="op">=</span><span class="dv">1</span>)</span>
    <span id="cb4-8"><a href="#cb4-8" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-9"><a href="#cb4-9" aria-hidden="true" tabindex="-1"></a><span class="co"># Set an xarray for all (time, amplitude) value combinations</span></span>
    <span id="cb4-10"><a href="#cb4-10" aria-hidden="true" tabindex="-1"></a>f1<span class="op">=</span> <span class="kw">lambda</span> t, T: np.mod(t<span class="op">/</span>T, <span class="dv">1</span>)</span>
    <span id="cb4-11"><a href="#cb4-11" aria-hidden="true" tabindex="-1"></a>tt1, TT <span class="op">=</span> np.meshgrid(t1, T1.val)</span>
    <span id="cb4-12"><a href="#cb4-12" aria-hidden="true" tabindex="-1"></a>y1 <span class="op">=</span> xr.DataArray(f1(tt1, TT),</span>
    <span id="cb4-13"><a href="#cb4-13" aria-hidden="true" tabindex="-1"></a>            dims<span class="op">=</span>[<span class="st">&#39;T&#39;</span>, <span class="st">&#39;t&#39;</span>],</span>
    <span id="cb4-14"><a href="#cb4-14" aria-hidden="true" tabindex="-1"></a>            coords<span class="op">=</span>{<span class="st">&#39;T&#39;</span>: T1.val, <span class="st">&#39;t&#39;</span>: t1})</span>
    <span id="cb4-15"><a href="#cb4-15" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-16"><a href="#cb4-16" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-17"><a href="#cb4-17" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the graph</span></span>
    <span id="cb4-18"><a href="#cb4-18" aria-hidden="true" tabindex="-1"></a>trace0 <span class="op">=</span> go.Scatter(x<span class="op">=</span>t1, </span>
    <span id="cb4-19"><a href="#cb4-19" aria-hidden="true" tabindex="-1"></a>                    y<span class="op">=</span>y1.sel(T<span class="op">=</span>T1.val_ini).values)</span>
    <span id="cb4-20"><a href="#cb4-20" aria-hidden="true" tabindex="-1"></a>fig1 <span class="op">=</span> go.FigureWidget([trace0])</span>
    <span id="cb4-21"><a href="#cb4-21" aria-hidden="true" tabindex="-1"></a>fig1.update_layout(template<span class="op">=</span><span class="st">&#39;none&#39;</span>,</span>
    <span id="cb4-22"><a href="#cb4-22" aria-hidden="true" tabindex="-1"></a>                    width<span class="op">=</span><span class="dv">800</span>, height<span class="op">=</span><span class="dv">500</span>,</span>
    <span id="cb4-23"><a href="#cb4-23" aria-hidden="true" tabindex="-1"></a>                    title<span class="op">=</span><span class="st">&quot;Flashes of a single firefly&quot;</span>,</span>
    <span id="cb4-24"><a href="#cb4-24" aria-hidden="true" tabindex="-1"></a>                    title_x<span class="op">=</span><span class="fl">0.5</span>,</span>
    <span id="cb4-25"><a href="#cb4-25" aria-hidden="true" tabindex="-1"></a>                    xaxis_title<span class="op">=</span><span class="st">&quot;t&quot;</span>,</span>
    <span id="cb4-26"><a href="#cb4-26" aria-hidden="true" tabindex="-1"></a>                    yaxis_title<span class="op">=</span><span class="st">&#39;θ&#39;</span>)</span>
    <span id="cb4-27"><a href="#cb4-27" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-28"><a href="#cb4-28" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-29"><a href="#cb4-29" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the callback to update the graph </span></span>
    <span id="cb4-30"><a href="#cb4-30" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> update1(change):</span>
    <span id="cb4-31"><a href="#cb4-31" aria-hidden="true" tabindex="-1"></a>    fig1.data[<span class="dv">0</span>].y <span class="op">=</span> y1.sel(T<span class="op">=</span>change[<span class="st">&#39;new&#39;</span>]).values</span>
    <span id="cb4-32"><a href="#cb4-32" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-33"><a href="#cb4-33" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-34"><a href="#cb4-34" aria-hidden="true" tabindex="-1"></a><span class="co"># Link the slider and the callback</span></span>
    <span id="cb4-35"><a href="#cb4-35" aria-hidden="true" tabindex="-1"></a>T1.slider.observe(update1, names<span class="op">=</span><span class="st">&#39;value&#39;</span>)</span>
    <span id="cb4-36"><a href="#cb4-36" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb4-37"><a href="#cb4-37" aria-hidden="true" tabindex="-1"></a><span class="co"># Display</span></span>
    <span id="cb4-38"><a href="#cb4-38" aria-hidden="true" tabindex="-1"></a>display(widgets.VBox([T1.slider, fig1]))</span></code></pre></div>
    <pre><code>VBox(children=(FloatSlider(value=2.0, description=&#39;period T&#39;, max=10.0, min=1.0, step=1.0), FigureWidget({
       …</code></pre>
    <h3 id="synchronization-process">Synchronization process</h3>
    <ul>
    <li>When a firefly flashes, it influences its neighbours</li>
    <li>The neighbours slows down or speeds up so as to flash more nearly in phase on the next cycle</li>
    <li>A simple model satisfying this hypothesis is:</li>
    </ul>
    <p><span class="math display">$$\begin{align*}\theta_{t+1} =  \theta_t + \frac{1}{T} + A\sin(\Theta_t - \theta_t)\end{align*}$$</span></p>
    <p>where <span class="math inline"><em>Θ</em><sub><em>t</em></sub></span> is the phase of a flashing (neigbhoring) firely[^1].</p>
    <p></p>
    <div class="sourceCode" id="cb6"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a><span class="co"># set the variable</span></span>
    <span id="cb6-2"><a href="#cb6-2" aria-hidden="true" tabindex="-1"></a>t2 <span class="op">=</span> np.linspace(<span class="dv">0</span>,<span class="dv">1</span>,<span class="dv">1000</span>)  <span class="co"># time array</span></span>
    <span id="cb6-3"><a href="#cb6-3" aria-hidden="true" tabindex="-1"></a>T2 <span class="op">=</span> <span class="dv">200</span>                    <span class="co"># the period</span></span>
    <span id="cb6-4"><a href="#cb6-4" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-5"><a href="#cb6-5" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-6"><a href="#cb6-6" aria-hidden="true" tabindex="-1"></a><span class="co"># Set a slider to check the influence of the resetting strength</span></span>
    <span id="cb6-7"><a href="#cb6-7" aria-hidden="true" tabindex="-1"></a>A <span class="op">=</span> Slider(name<span class="op">=</span><span class="st">&#39;Amplitude&#39;</span>, start<span class="op">=</span><span class="dv">0</span>, stop<span class="op">=</span><span class="dv">1</span>, step<span class="op">=</span><span class="fl">0.01</span>)</span>
    <span id="cb6-8"><a href="#cb6-8" aria-hidden="true" tabindex="-1"></a>A.slider.description_tooltip <span class="op">=</span> <span class="st">&quot;Amplitude of the resetting strength</span><span class="ch">\n</span><span class="st"> &quot;</span>\</span>
    <span id="cb6-9"><a href="#cb6-9" aria-hidden="true" tabindex="-1"></a>                                <span class="st">&quot;(it measures the firefly’s ability to &quot;</span>\</span>
    <span id="cb6-10"><a href="#cb6-10" aria-hidden="true" tabindex="-1"></a>                                <span class="st">&quot;modify its instantaneous frequency)&quot;</span></span>
    <span id="cb6-11"><a href="#cb6-11" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-12"><a href="#cb6-12" aria-hidden="true" tabindex="-1"></a><span class="co"># Set an xarray for all (time, amplitude) value combinations</span></span>
    <span id="cb6-13"><a href="#cb6-13" aria-hidden="true" tabindex="-1"></a><span class="co"># Option 1: using modulus</span></span>
    <span id="cb6-14"><a href="#cb6-14" aria-hidden="true" tabindex="-1"></a>f2<span class="op">=</span> <span class="kw">lambda</span> t, A: np.mod(t <span class="op">+</span> <span class="dv">1</span><span class="op">/</span>T2 <span class="op">+</span> A<span class="op">*</span>np.sin(<span class="dv">2</span><span class="op">*</span>np.pi<span class="op">*</span>(<span class="fl">1.005</span><span class="op">-</span>t)), <span class="dv">1</span>)</span>
    <span id="cb6-15"><a href="#cb6-15" aria-hidden="true" tabindex="-1"></a>tt2, AA <span class="op">=</span> np.meshgrid(t2, A.val)</span>
    <span id="cb6-16"><a href="#cb6-16" aria-hidden="true" tabindex="-1"></a>y2 <span class="op">=</span> xr.DataArray(f2(tt2, AA),</span>
    <span id="cb6-17"><a href="#cb6-17" aria-hidden="true" tabindex="-1"></a>            dims<span class="op">=</span>[<span class="st">&#39;Amplitude&#39;</span>, <span class="st">&#39;t&#39;</span>],</span>
    <span id="cb6-18"><a href="#cb6-18" aria-hidden="true" tabindex="-1"></a>            coords<span class="op">=</span>{<span class="st">&#39;Amplitude&#39;</span>: A.val, <span class="st">&#39;t&#39;</span>: t2})</span>
    <span id="cb6-19"><a href="#cb6-19" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-20"><a href="#cb6-20" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-21"><a href="#cb6-21" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-22"><a href="#cb6-22" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the graph</span></span>
    <span id="cb6-23"><a href="#cb6-23" aria-hidden="true" tabindex="-1"></a>trace0 <span class="op">=</span> go.Scatter(x<span class="op">=</span>t2, </span>
    <span id="cb6-24"><a href="#cb6-24" aria-hidden="true" tabindex="-1"></a>                    y<span class="op">=</span>y2.sel(Amplitude<span class="op">=</span>A.val_ini).values,</span>
    <span id="cb6-25"><a href="#cb6-25" aria-hidden="true" tabindex="-1"></a>                    name<span class="op">=</span><span class="st">&#39;with coupling&#39;</span>)</span>
    <span id="cb6-26"><a href="#cb6-26" aria-hidden="true" tabindex="-1"></a>trace1 <span class="op">=</span> go.Scatter(x<span class="op">=</span>t2,</span>
    <span id="cb6-27"><a href="#cb6-27" aria-hidden="true" tabindex="-1"></a>                    y<span class="op">=</span>np.mod(t2<span class="op">+</span>(<span class="dv">1</span><span class="op">/</span>T2),<span class="dv">1</span>),</span>
    <span id="cb6-28"><a href="#cb6-28" aria-hidden="true" tabindex="-1"></a>                    name<span class="op">=</span><span class="st">&#39;no coupling&#39;</span>)</span>
    <span id="cb6-29"><a href="#cb6-29" aria-hidden="true" tabindex="-1"></a>fig2 <span class="op">=</span> go.FigureWidget([trace0, trace1])</span>
    <span id="cb6-30"><a href="#cb6-30" aria-hidden="true" tabindex="-1"></a>fig2.update_layout(template<span class="op">=</span><span class="st">&#39;none&#39;</span>,</span>
    <span id="cb6-31"><a href="#cb6-31" aria-hidden="true" tabindex="-1"></a>                    width<span class="op">=</span><span class="dv">800</span>, height<span class="op">=</span><span class="dv">500</span>,</span>
    <span id="cb6-32"><a href="#cb6-32" aria-hidden="true" tabindex="-1"></a>                    title<span class="op">=</span><span class="st">&quot;Influence of the resetting strength&quot;</span>,</span>
    <span id="cb6-33"><a href="#cb6-33" aria-hidden="true" tabindex="-1"></a>                    title_x<span class="op">=</span><span class="fl">0.5</span>,</span>
    <span id="cb6-34"><a href="#cb6-34" aria-hidden="true" tabindex="-1"></a>                    xaxis_title<span class="op">=</span><span class="st">&quot;θ&lt;sub&gt;t&lt;sub&gt;&quot;</span>,</span>
    <span id="cb6-35"><a href="#cb6-35" aria-hidden="true" tabindex="-1"></a>                    yaxis_title<span class="op">=</span><span class="st">&quot;θ&lt;sub&gt;t+1&lt;sub&gt;&quot;</span>,</span>
    <span id="cb6-36"><a href="#cb6-36" aria-hidden="true" tabindex="-1"></a>                    legend_title<span class="op">=</span><span class="st">&#39;Click to deselect&#39;</span>, </span>
    <span id="cb6-37"><a href="#cb6-37" aria-hidden="true" tabindex="-1"></a>                    legend_title_font<span class="op">=</span><span class="bu">dict</span>(size<span class="op">=</span><span class="dv">16</span>),</span>
    <span id="cb6-38"><a href="#cb6-38" aria-hidden="true" tabindex="-1"></a>                    legend_title_font_color<span class="op">=</span><span class="st">&#39;FireBrick&#39;</span>)</span>
    <span id="cb6-39"><a href="#cb6-39" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-40"><a href="#cb6-40" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-41"><a href="#cb6-41" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the callback to update the graph </span></span>
    <span id="cb6-42"><a href="#cb6-42" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> update2(change):</span>
    <span id="cb6-43"><a href="#cb6-43" aria-hidden="true" tabindex="-1"></a>    <span class="cf">with</span> fig2.batch_update():</span>
    <span id="cb6-44"><a href="#cb6-44" aria-hidden="true" tabindex="-1"></a>        fig2.data[<span class="dv">0</span>].y <span class="op">=</span> y2.sel(Amplitude<span class="op">=</span>change[<span class="st">&#39;new&#39;</span>]).values</span>
    <span id="cb6-45"><a href="#cb6-45" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-46"><a href="#cb6-46" aria-hidden="true" tabindex="-1"></a><span class="co"># Link the slider and the callback</span></span>
    <span id="cb6-47"><a href="#cb6-47" aria-hidden="true" tabindex="-1"></a>A.slider.observe(update2, names<span class="op">=</span><span class="st">&#39;value&#39;</span>)</span>
    <span id="cb6-48"><a href="#cb6-48" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-49"><a href="#cb6-49" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb6-50"><a href="#cb6-50" aria-hidden="true" tabindex="-1"></a><span class="co"># Display</span></span>
    <span id="cb6-51"><a href="#cb6-51" aria-hidden="true" tabindex="-1"></a>display(widgets.VBox([A.slider, fig2]))</span></code></pre></div>
    <pre><code>VBox(children=(FloatSlider(value=0.08, description=&#39;Amplitude&#39;, description_tooltip=&#39;Amplitude of the resettin…</code></pre>
    <h2 id="cellular-automata-modeling">Cellular Automata modeling</h2>
    <h3 id="parameters-of-the-simulation">Parameters of the simulation</h3>
    <div class="sourceCode" id="cb8"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb8-1"><a href="#cb8-1" aria-hidden="true" tabindex="-1"></a><span class="at">@dataclass</span></span>
    <span id="cb8-2"><a href="#cb8-2" aria-hidden="true" tabindex="-1"></a><span class="kw">class</span> Parameters:</span>
    <span id="cb8-3"><a href="#cb8-3" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;This class contains the parameters of the simulation.</span></span>
    <span id="cb8-4"><a href="#cb8-4" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb8-5"><a href="#cb8-5" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb8-6"><a href="#cb8-6" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Size of the grid</span></span>
    <span id="cb8-7"><a href="#cb8-7" aria-hidden="true" tabindex="-1"></a>    M: <span class="bu">int</span> <span class="op">=</span> <span class="dv">10</span>            <span class="co"># number of cells in the x direction</span></span>
    <span id="cb8-8"><a href="#cb8-8" aria-hidden="true" tabindex="-1"></a>    N: <span class="bu">int</span> <span class="op">=</span> <span class="dv">10</span>            <span class="co"># number of cells in the y direction</span></span>
    <span id="cb8-9"><a href="#cb8-9" aria-hidden="true" tabindex="-1"></a>    fireflies_density: <span class="bu">float</span> <span class="op">=</span> <span class="dv">1</span>  <span class="co"># how much of the grid is populated</span></span>
    <span id="cb8-10"><a href="#cb8-10" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb8-11"><a href="#cb8-11" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Coupling parameters</span></span>
    <span id="cb8-12"><a href="#cb8-12" aria-hidden="true" tabindex="-1"></a>    coupling_value: <span class="bu">float</span> <span class="op">=</span> <span class="fl">0.1</span>    <span class="co"># float (between 0 and +/- 0.3)</span></span>
    <span id="cb8-13"><a href="#cb8-13" aria-hidden="true" tabindex="-1"></a>    neighbor_distance: <span class="bu">float</span> <span class="op">=</span> <span class="dv">3</span>  <span class="co"># Size of neighbourhood radius</span></span>
    <span id="cb8-14"><a href="#cb8-14" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb8-15"><a href="#cb8-15" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Simulation settings</span></span>
    <span id="cb8-16"><a href="#cb8-16" aria-hidden="true" tabindex="-1"></a>    nb_periods: <span class="bu">int</span> <span class="op">=</span> <span class="dv">20</span></span>
    <span id="cb8-17"><a href="#cb8-17" aria-hidden="true" tabindex="-1"></a>    time_step_per_period: <span class="bu">int</span> <span class="op">=</span> <span class="dv">100</span>  <span class="co"># dt = 1/time_step_per_period</span></span></code></pre></div>
    <h3 id="simulation-script">Simulation script</h3>
    <div class="sourceCode" id="cb9"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb9-1"><a href="#cb9-1" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> fireflies_simulator(par):</span>
    <span id="cb9-2"><a href="#cb9-2" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;This function simulates the fireflies system, computes the order</span></span>
    <span id="cb9-3"><a href="#cb9-3" aria-hidden="true" tabindex="-1"></a><span class="co">    parameter and shows figures of the system state during the simulation.</span></span>
    <span id="cb9-4"><a href="#cb9-4" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-5"><a href="#cb9-5" aria-hidden="true" tabindex="-1"></a><span class="co">    Args:</span></span>
    <span id="cb9-6"><a href="#cb9-6" aria-hidden="true" tabindex="-1"></a><span class="co">        - Parameters</span></span>
    <span id="cb9-7"><a href="#cb9-7" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-8"><a href="#cb9-8" aria-hidden="true" tabindex="-1"></a><span class="co">    Returns:</span></span>
    <span id="cb9-9"><a href="#cb9-9" aria-hidden="true" tabindex="-1"></a><span class="co">        The value of the order parameter over time</span></span>
    <span id="cb9-10"><a href="#cb9-10" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb9-11"><a href="#cb9-11" aria-hidden="true" tabindex="-1"></a>    time_steps <span class="op">=</span> np.arange(start<span class="op">=</span><span class="dv">0</span>,</span>
    <span id="cb9-12"><a href="#cb9-12" aria-hidden="true" tabindex="-1"></a>                            stop<span class="op">=</span>par.nb_periods <span class="op">*</span> par.time_step_per_period,</span>
    <span id="cb9-13"><a href="#cb9-13" aria-hidden="true" tabindex="-1"></a>                            step<span class="op">=</span><span class="dv">1</span>)</span>
    <span id="cb9-14"><a href="#cb9-14" aria-hidden="true" tabindex="-1"></a>    phases <span class="op">=</span> np.zeros((time_steps.size<span class="op">+</span><span class="dv">1</span>, par.N, par.M))  <span class="co"># 1 grid (MxN) per time step</span></span>
    <span id="cb9-15"><a href="#cb9-15" aria-hidden="true" tabindex="-1"></a>    phases[<span class="dv">0</span>] <span class="op">=</span> np.random.random((par.N, par.M))          <span class="co"># random initial state</span></span>
    <span id="cb9-16"><a href="#cb9-16" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-17"><a href="#cb9-17" aria-hidden="true" tabindex="-1"></a>    <span class="co"># empty some cells to ensure the right fireflies density</span></span>
    <span id="cb9-18"><a href="#cb9-18" aria-hidden="true" tabindex="-1"></a>    nb_empty_cell <span class="op">=</span> np.around((<span class="dv">1</span><span class="op">-</span>par.fireflies_density)<span class="op">*</span>par.N<span class="op">*</span>par.M, <span class="dv">0</span>).astype(<span class="bu">int</span>)</span>
    <span id="cb9-19"><a href="#cb9-19" aria-hidden="true" tabindex="-1"></a>    ind <span class="op">=</span> np.random.choice(phases[<span class="dv">0</span>].size, size<span class="op">=</span>nb_empty_cell, replace<span class="op">=</span><span class="va">False</span>)</span>
    <span id="cb9-20"><a href="#cb9-20" aria-hidden="true" tabindex="-1"></a>    phases[<span class="dv">0</span>].ravel()[ind] <span class="op">=</span> np.nan</span>
    <span id="cb9-21"><a href="#cb9-21" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-22"><a href="#cb9-22" aria-hidden="true" tabindex="-1"></a>    neighbors <span class="op">=</span> compute_neighborhood(phases[<span class="dv">0</span>], par.N, par.M, par.neighbor_distance)</span>
    <span id="cb9-23"><a href="#cb9-23" aria-hidden="true" tabindex="-1"></a>    phase_increment <span class="op">=</span> <span class="dv">1</span> <span class="op">/</span> par.time_step_per_period</span>
    <span id="cb9-24"><a href="#cb9-24" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-25"><a href="#cb9-25" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> t <span class="kw">in</span> time_steps:</span>
    <span id="cb9-26"><a href="#cb9-26" aria-hidden="true" tabindex="-1"></a>        phases[t<span class="op">+</span><span class="dv">1</span>] <span class="op">=</span> phases[t] <span class="op">+</span> phase_increment</span>
    <span id="cb9-27"><a href="#cb9-27" aria-hidden="true" tabindex="-1"></a>        glow_idx <span class="op">=</span> np.array(np.nonzero(phases[t<span class="op">+</span><span class="dv">1</span>]<span class="op">&gt;</span><span class="dv">1</span>)).T</span>
    <span id="cb9-28"><a href="#cb9-28" aria-hidden="true" tabindex="-1"></a>        ids <span class="op">=</span> [np.ravel_multi_index(tup, (par.N, par.M)) <span class="cf">for</span> tup <span class="kw">in</span> glow_idx]</span>
    <span id="cb9-29"><a href="#cb9-29" aria-hidden="true" tabindex="-1"></a>        <span class="cf">for</span> ind <span class="kw">in</span> ids:</span>
    <span id="cb9-30"><a href="#cb9-30" aria-hidden="true" tabindex="-1"></a>            i,j <span class="op">=</span> np.unravel_index(ind, (par.N, par.M))</span>
    <span id="cb9-31"><a href="#cb9-31" aria-hidden="true" tabindex="-1"></a>            phases[t<span class="op">+</span><span class="dv">1</span>][neighbors[ind]] <span class="op">=</span> nudge(phases[t<span class="op">+</span><span class="dv">1</span>][neighbors[ind]], </span>
    <span id="cb9-32"><a href="#cb9-32" aria-hidden="true" tabindex="-1"></a>                                                phases[t<span class="op">+</span><span class="dv">1</span>][i,j],</span>
    <span id="cb9-33"><a href="#cb9-33" aria-hidden="true" tabindex="-1"></a>                                                par.coupling_value)</span>
    <span id="cb9-34"><a href="#cb9-34" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-35"><a href="#cb9-35" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span>  phases</span>
    <span id="cb9-36"><a href="#cb9-36" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-37"><a href="#cb9-37" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-38"><a href="#cb9-38" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> compute_neighborhood(grid, N, M, r):</span>
    <span id="cb9-39"><a href="#cb9-39" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;For every fireflies, compute a mask array of neighboring fireflies.</span></span>
    <span id="cb9-40"><a href="#cb9-40" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-41"><a href="#cb9-41" aria-hidden="true" tabindex="-1"></a><span class="co">    Args:</span></span>
    <span id="cb9-42"><a href="#cb9-42" aria-hidden="true" tabindex="-1"></a><span class="co">        grid (ndarray): phase state of each firefly</span></span>
    <span id="cb9-43"><a href="#cb9-43" aria-hidden="true" tabindex="-1"></a><span class="co">        N (int): number of cells in the x direction</span></span>
    <span id="cb9-44"><a href="#cb9-44" aria-hidden="true" tabindex="-1"></a><span class="co">        M (int): number of cells in the y direction</span></span>
    <span id="cb9-45"><a href="#cb9-45" aria-hidden="true" tabindex="-1"></a><span class="co">        r (float): radius of neighbour_distance</span></span>
    <span id="cb9-46"><a href="#cb9-46" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-47"><a href="#cb9-47" aria-hidden="true" tabindex="-1"></a><span class="co">    Returns:</span></span>
    <span id="cb9-48"><a href="#cb9-48" aria-hidden="true" tabindex="-1"></a><span class="co">        dict: keys: index (ravel) of each firefly</span></span>
    <span id="cb9-49"><a href="#cb9-49" aria-hidden="true" tabindex="-1"></a><span class="co">                values: mask array with neighbour fireflies</span></span>
    <span id="cb9-50"><a href="#cb9-50" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb9-51"><a href="#cb9-51" aria-hidden="true" tabindex="-1"></a>    neighbors <span class="op">=</span> <span class="bu">dict</span>.fromkeys((<span class="bu">range</span>(N<span class="op">*</span>M)), [])</span>
    <span id="cb9-52"><a href="#cb9-52" aria-hidden="true" tabindex="-1"></a>    occupied_cells <span class="op">=</span> <span class="op">~</span>np.isnan(grid)</span>
    <span id="cb9-53"><a href="#cb9-53" aria-hidden="true" tabindex="-1"></a>    <span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(N):</span>
    <span id="cb9-54"><a href="#cb9-54" aria-hidden="true" tabindex="-1"></a>        <span class="cf">for</span> j <span class="kw">in</span> <span class="bu">range</span>(M):</span>
    <span id="cb9-55"><a href="#cb9-55" aria-hidden="true" tabindex="-1"></a>            <span class="cf">if</span> np.isnan(grid[i,j]):</span>
    <span id="cb9-56"><a href="#cb9-56" aria-hidden="true" tabindex="-1"></a>                <span class="cf">continue</span></span>
    <span id="cb9-57"><a href="#cb9-57" aria-hidden="true" tabindex="-1"></a>            y, x <span class="op">=</span> np.ogrid[<span class="op">-</span>i:N<span class="op">-</span>i, <span class="op">-</span>j:M<span class="op">-</span>j]</span>
    <span id="cb9-58"><a href="#cb9-58" aria-hidden="true" tabindex="-1"></a>            mask <span class="op">=</span> (x<span class="op">**</span><span class="dv">2</span> <span class="op">+</span> y<span class="op">**</span><span class="dv">2</span> <span class="op">&lt;=</span> r<span class="op">**</span><span class="dv">2</span>)<span class="op">*</span>occupied_cells  <span class="co"># only keep occupied neighboring cells</span></span>
    <span id="cb9-59"><a href="#cb9-59" aria-hidden="true" tabindex="-1"></a>            <span class="co"># mask[i,j] = False                            # don&#39;t include the cell istelf</span></span>
    <span id="cb9-60"><a href="#cb9-60" aria-hidden="true" tabindex="-1"></a>            neighbors[np.ravel_multi_index((i,j), (N,M))] <span class="op">=</span> mask</span>
    <span id="cb9-61"><a href="#cb9-61" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> neighbors</span>
    <span id="cb9-62"><a href="#cb9-62" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-63"><a href="#cb9-63" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-64"><a href="#cb9-64" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> nudge(neighbor_phases, flash_phase, amplitude):</span>
    <span id="cb9-65"><a href="#cb9-65" aria-hidden="true" tabindex="-1"></a>    <span class="co">&quot;&quot;&quot;Nudge the neighboring fireflies.</span></span>
    <span id="cb9-66"><a href="#cb9-66" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb9-67"><a href="#cb9-67" aria-hidden="true" tabindex="-1"></a><span class="co">    Args:</span></span>
    <span id="cb9-68"><a href="#cb9-68" aria-hidden="true" tabindex="-1"></a><span class="co">        neighbor_phases (ndarray): phases of all the fireflies to nudge</span></span>
    <span id="cb9-69"><a href="#cb9-69" aria-hidden="true" tabindex="-1"></a><span class="co">        flash_phase (float): phase of the flashing firefly</span></span>
    <span id="cb9-70"><a href="#cb9-70" aria-hidden="true" tabindex="-1"></a><span class="co">        amplitude (float): resetting strength / coupling value</span></span>
    <span id="cb9-71"><a href="#cb9-71" aria-hidden="true" tabindex="-1"></a><span class="co">    &quot;&quot;&quot;</span></span>
    <span id="cb9-72"><a href="#cb9-72" aria-hidden="true" tabindex="-1"></a>    phase_diff <span class="op">=</span> flash_phase <span class="op">-</span> neighbor_phases</span>
    <span id="cb9-73"><a href="#cb9-73" aria-hidden="true" tabindex="-1"></a>    res <span class="op">=</span> neighbor_phases <span class="op">+</span> amplitude<span class="op">*</span>np.sin(<span class="dv">2</span><span class="op">*</span>np.pi<span class="op">*</span>(phase_diff))</span>
    <span id="cb9-74"><a href="#cb9-74" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> np.mod(res, <span class="dv">1</span>)</span></code></pre></div>
    <h3 id="simulation">Simulation</h3>
    <div class="sourceCode" id="cb10"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb10-1"><a href="#cb10-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Let&#39;s initiate the (default) parameters for a simulation</span></span>
    <span id="cb10-2"><a href="#cb10-2" aria-hidden="true" tabindex="-1"></a>settings <span class="op">=</span> Parameters(N<span class="op">=</span><span class="dv">20</span>, M<span class="op">=</span><span class="dv">20</span>, </span>
    <span id="cb10-3"><a href="#cb10-3" aria-hidden="true" tabindex="-1"></a>                    fireflies_density<span class="op">=</span><span class="fl">0.8</span>, </span>
    <span id="cb10-4"><a href="#cb10-4" aria-hidden="true" tabindex="-1"></a>                    coupling_value<span class="op">=</span><span class="fl">0.01</span>, </span>
    <span id="cb10-5"><a href="#cb10-5" aria-hidden="true" tabindex="-1"></a>                    neighbor_distance<span class="op">=</span> <span class="dv">2</span>,</span>
    <span id="cb10-6"><a href="#cb10-6" aria-hidden="true" tabindex="-1"></a>                    time_step_per_period<span class="op">=</span><span class="dv">100</span>, </span>
    <span id="cb10-7"><a href="#cb10-7" aria-hidden="true" tabindex="-1"></a>                    nb_periods<span class="op">=</span><span class="dv">100</span></span>
    <span id="cb10-8"><a href="#cb10-8" aria-hidden="true" tabindex="-1"></a>)</span>
    <span id="cb10-9"><a href="#cb10-9" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb10-10"><a href="#cb10-10" aria-hidden="true" tabindex="-1"></a><span class="co"># And let&#39;s run the simulation</span></span>
    <span id="cb10-11"><a href="#cb10-11" aria-hidden="true" tabindex="-1"></a>heatmaps <span class="op">=</span> fireflies_simulator(settings)</span></code></pre></div>
    <div class="sourceCode" id="cb11"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb11-1"><a href="#cb11-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Let&#39;s visualiza the simulation</span></span>
    <span id="cb11-2"><a href="#cb11-2" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-3"><a href="#cb11-3" aria-hidden="true" tabindex="-1"></a>mymin <span class="op">=</span> <span class="dv">0</span></span>
    <span id="cb11-4"><a href="#cb11-4" aria-hidden="true" tabindex="-1"></a>mymax <span class="op">=</span> heatmaps.shape[<span class="dv">0</span>]<span class="op">-</span><span class="dv">1</span></span>
    <span id="cb11-5"><a href="#cb11-5" aria-hidden="true" tabindex="-1"></a>mystep <span class="op">=</span> <span class="dv">10</span></span>
    <span id="cb11-6"><a href="#cb11-6" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-7"><a href="#cb11-7" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-8"><a href="#cb11-8" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the figure</span></span>
    <span id="cb11-9"><a href="#cb11-9" aria-hidden="true" tabindex="-1"></a>fig5 <span class="op">=</span> go.FigureWidget(</span>
    <span id="cb11-10"><a href="#cb11-10" aria-hidden="true" tabindex="-1"></a>    data<span class="op">=</span>go.Heatmap(z<span class="op">=</span>heatmaps[<span class="dv">0</span>], colorscale<span class="op">=</span><span class="st">&#39;Hot&#39;</span>, reversescale<span class="op">=</span><span class="va">True</span>),</span>
    <span id="cb11-11"><a href="#cb11-11" aria-hidden="true" tabindex="-1"></a>    layout<span class="op">=</span>go.Layout(title<span class="op">=</span><span class="st">&quot;Fireflies Simulator&quot;</span>))</span>
    <span id="cb11-12"><a href="#cb11-12" aria-hidden="true" tabindex="-1"></a>fig5.data[<span class="dv">0</span>].update(zmin<span class="op">=</span><span class="dv">0</span>, zmax<span class="op">=</span><span class="dv">1</span>)  <span class="co"># fix the colorscale to [0,1]</span></span>
    <span id="cb11-13"><a href="#cb11-13" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-14"><a href="#cb11-14" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-15"><a href="#cb11-15" aria-hidden="true" tabindex="-1"></a><span class="co"># Set the callback to update the graph </span></span>
    <span id="cb11-16"><a href="#cb11-16" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> update5(change):</span>
    <span id="cb11-17"><a href="#cb11-17" aria-hidden="true" tabindex="-1"></a>    fig5.data[<span class="dv">0</span>].z <span class="op">=</span> heatmaps[change[<span class="st">&#39;new&#39;</span>]]</span>
    <span id="cb11-18"><a href="#cb11-18" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-19"><a href="#cb11-19" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-20"><a href="#cb11-20" aria-hidden="true" tabindex="-1"></a><span class="co"># Create the animation display and link it to the callback function</span></span>
    <span id="cb11-21"><a href="#cb11-21" aria-hidden="true" tabindex="-1"></a>myanimation <span class="op">=</span> AnimateSlider(start<span class="op">=</span>mymin, stop<span class="op">=</span>mymax, step<span class="op">=</span>mystep)</span>
    <span id="cb11-22"><a href="#cb11-22" aria-hidden="true" tabindex="-1"></a>myanimation.slider.observe(update5, names<span class="op">=</span><span class="st">&#39;value&#39;</span>)</span>
    <span id="cb11-23"><a href="#cb11-23" aria-hidden="true" tabindex="-1"></a>controllers <span class="op">=</span> widgets.HBox([myanimation.play, myanimation.slider])</span>
    <span id="cb11-24"><a href="#cb11-24" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-25"><a href="#cb11-25" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb11-26"><a href="#cb11-26" aria-hidden="true" tabindex="-1"></a><span class="co"># Display</span></span>
    <span id="cb11-27"><a href="#cb11-27" aria-hidden="true" tabindex="-1"></a>display(widgets.VBox([controllers, fig5]))</span></code></pre></div>
    <pre><code>VBox(children=(HBox(children=(Play(value=0, description=&#39;Press play&#39;, max=10000, step=10), IntSlider(value=0, …</code></pre>
    <div class="sourceCode" id="cb13"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb13-1"><a href="#cb13-1" aria-hidden="true" tabindex="-1"></a><span class="co"># Statistical results</span></span>
    <span id="cb13-2"><a href="#cb13-2" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb13-3"><a href="#cb13-3" aria-hidden="true" tabindex="-1"></a>t3 <span class="op">=</span> np.arange(heatmaps.shape[<span class="dv">0</span>])</span>
    <span id="cb13-4"><a href="#cb13-4" aria-hidden="true" tabindex="-1"></a>avg <span class="op">=</span> np.nanmean(heatmaps, axis<span class="op">=</span>(<span class="dv">1</span>,<span class="dv">2</span>))</span>
    <span id="cb13-5"><a href="#cb13-5" aria-hidden="true" tabindex="-1"></a>std <span class="op">=</span> np.nanstd(heatmaps, axis<span class="op">=</span>(<span class="dv">1</span>,<span class="dv">2</span>))</span></code></pre></div>
    <div class="sourceCode" id="cb14"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb14-1"><a href="#cb14-1" aria-hidden="true" tabindex="-1"></a><span class="co">#show light intensity over time (the last 500 time steps)</span></span>
    <span id="cb14-2"><a href="#cb14-2" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb14-3"><a href="#cb14-3" aria-hidden="true" tabindex="-1"></a>trace0 <span class="op">=</span> go.Scatter(x<span class="op">=</span>t3[<span class="op">-</span><span class="dv">1000</span>:], y<span class="op">=</span>avg[<span class="op">-</span><span class="dv">1000</span>:], name<span class="op">=</span><span class="st">&#39;average&#39;</span>)</span>
    <span id="cb14-4"><a href="#cb14-4" aria-hidden="true" tabindex="-1"></a>trace1 <span class="op">=</span> go.Scatter(x<span class="op">=</span>t3[<span class="op">-</span><span class="dv">1000</span>:], y<span class="op">=</span>std[<span class="op">-</span><span class="dv">1000</span>:], name<span class="op">=</span><span class="st">&#39;std dev&#39;</span>)</span>
    <span id="cb14-5"><a href="#cb14-5" aria-hidden="true" tabindex="-1"></a></span>
    <span id="cb14-6"><a href="#cb14-6" aria-hidden="true" tabindex="-1"></a>fig3 <span class="op">=</span> go.FigureWidget([trace0, trace1])</span>
    <span id="cb14-7"><a href="#cb14-7" aria-hidden="true" tabindex="-1"></a>fig3.update_yaxes(<span class="bu">range</span><span class="op">=</span>[<span class="dv">0</span>, <span class="dv">1</span>])       <span class="co"># fix the scale to [0,1] for easy comparison</span></span>
    <span id="cb14-8"><a href="#cb14-8" aria-hidden="true" tabindex="-1"></a>fig3.update_layout(template<span class="op">=</span><span class="st">&#39;none&#39;</span>,</span>
    <span id="cb14-9"><a href="#cb14-9" aria-hidden="true" tabindex="-1"></a>                    width<span class="op">=</span><span class="dv">800</span>, height<span class="op">=</span><span class="dv">500</span>,</span>
    <span id="cb14-10"><a href="#cb14-10" aria-hidden="true" tabindex="-1"></a>                    title<span class="op">=</span><span class="st">&quot;Light intensity&quot;</span>,</span>
    <span id="cb14-11"><a href="#cb14-11" aria-hidden="true" tabindex="-1"></a>                    title_x<span class="op">=</span><span class="fl">0.5</span>,</span>
    <span id="cb14-12"><a href="#cb14-12" aria-hidden="true" tabindex="-1"></a>                    xaxis_title<span class="op">=</span><span class="st">&quot;t&quot;</span>,</span>
    <span id="cb14-13"><a href="#cb14-13" aria-hidden="true" tabindex="-1"></a>                    yaxis_title<span class="op">=</span><span class="st">&quot;I&quot;</span>)</span>
    <span id="cb14-14"><a href="#cb14-14" aria-hidden="true" tabindex="-1"></a>fig3</span></code></pre></div>
    <pre><code>FigureWidget({
        &#39;data&#39;: [{&#39;name&#39;: &#39;average&#39;,
                  &#39;type&#39;: &#39;scatter&#39;,
                  &#39;uid&#39;: &#39;f855cbb…</code></pre>
    <!--bibtex
    
    @Article{PER-GRA:2007,
      Author    = {P\'erez, Fernando and Granger, Brian E.},
      Title     = {{IP}ython: a System for Interactive Scientific Computing},
      Journal   = {Computing in Science and Engineering},
      Volume    = {9},
      Number    = {3},
      Pages     = {21--29},
      month     = may,
      year      = 2007,
      url       = "http://ipython.org",
      ISSN      = "1521-9615",
      doi       = {10.1109/MCSE.2007.53},
      publisher = {IEEE Computer Society},
    }
    
    @article{Papa2007,
      author = {Papa, David A. and Markov, Igor L.},
      journal = {Approximation algorithms and metaheuristics},
      pages = {1--38},
      title = {{Hypergraph partitioning and clustering}},
      url = {http://www.podload.org/pubs/book/part\_survey.pdf},
      year = {2007}
    }
    
    -->
    <p>Examples of citations: <a href="#cite-PER-GRA:2007">CITE</a> or <a href="#cite-Papa2007">CITE</a>.</p>
    <!--
    
    
    ```python
    #automatic document conversion to markdown and then to word
    #first convert the ipython notebook paper.ipynb to markdown
    os.system("jupyter nbconvert --to markdown Fireflies-Synchronization.ipynb")
        
     #next convert markdown to ms word
    conversion = f"pandoc -s Fireflies-Synchronization.md --citeproc --bibliography Fireflies-Synchronization.bib --csl=apa.csl"
    
    os.system(conversion)
    ```
    
        [NbConvertApp] Converting notebook Fireflies-Synchronization.ipynb to markdown
        [NbConvertApp] Writing 17535 bytes to Fireflies-Synchronization.md
    
    
        <!DOCTYPE html>
        <html xmlns="http://www.w3.org/1999/xhtml" lang="" xml:lang="">
        <head>
          <meta charset="utf-8" />
          <meta name="generator" content="pandoc" />
          <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes" />
          <title>Fireflies-Synchronization</title>
          <style>
            html {
              line-height: 1.5;
              font-family: Georgia, serif;
              font-size: 20px;
              color: #1a1a1a;
              background-color: #fdfdfd;
            }
            body {
              margin: 0 auto;
              max-width: 36em;
              padding-left: 50px;
              padding-right: 50px;
              padding-top: 50px;
              padding-bottom: 50px;
              hyphens: auto;
              overflow-wrap: break-word;
              text-rendering: optimizeLegibility;
              font-kerning: normal;
            }
            @media (max-width: 600px) {
              body {
                font-size: 0.9em;
                padding: 1em;
              }
            }
            @media print {
              body {
                background-color: transparent;
                color: black;
                font-size: 12pt;
              }
              p, h2, h3 {
                orphans: 3;
                widows: 3;
              }
              h2, h3, h4 {
                page-break-after: avoid;
              }
            }
            p {
              margin: 1em 0;
            }
            a {
              color: #1a1a1a;
            }
            a:visited {
              color: #1a1a1a;
            }
            img {
              max-width: 100%;
            }
            h1, h2, h3, h4, h5, h6 {
              margin-top: 1.4em;
            }
            h5, h6 {
              font-size: 1em;
              font-style: italic;
            }
            h6 {
              font-weight: normal;
            }
            ol, ul {
              padding-left: 1.7em;
              margin-top: 1em;
            }
            li > ol, li > ul {
              margin-top: 0;
            }
            blockquote {
              margin: 1em 0 1em 1.7em;
              padding-left: 1em;
              border-left: 2px solid #e6e6e6;
              color: #606060;
            }
            code {
              font-family: Menlo, Monaco, 'Lucida Console', Consolas, monospace;
              font-size: 85%;
              margin: 0;
            }
            pre {
              margin: 1em 0;
              overflow: auto;
            }
            pre code {
              padding: 0;
              overflow: visible;
              overflow-wrap: normal;
            }
            .sourceCode {
             background-color: transparent;
             overflow: visible;
            }
            hr {
              background-color: #1a1a1a;
              border: none;
              height: 1px;
              margin: 1em 0;
            }
            table {
              margin: 1em 0;
              border-collapse: collapse;
              width: 100%;
              overflow-x: auto;
              display: block;
              font-variant-numeric: lining-nums tabular-nums;
            }
            table caption {
              margin-bottom: 0.75em;
            }
            tbody {
              margin-top: 0.5em;
              border-top: 1px solid #1a1a1a;
              border-bottom: 1px solid #1a1a1a;
            }
            th {
              border-top: 1px solid #1a1a1a;
              padding: 0.25em 0.5em 0.25em 0.5em;
            }
            td {
              padding: 0.125em 0.5em 0.25em 0.5em;
            }
            header {
              margin-bottom: 4em;
              text-align: center;
            }
            #TOC li {
              list-style: none;
            }
            #TOC a:not(:hover) {
              text-decoration: none;
            }
            code{white-space: pre-wrap;}
            span.smallcaps{font-variant: small-caps;}
            span.underline{text-decoration: underline;}
            div.column{display: inline-block; vertical-align: top; width: 50%;}
            div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;}
            ul.task-list{list-style: none;}
            pre > code.sourceCode { white-space: pre; position: relative; }
            pre > code.sourceCode > span { display: inline-block; line-height: 1.25; }
            pre > code.sourceCode > span:empty { height: 1.2em; }
            .sourceCode { overflow: visible; }
            code.sourceCode > span { color: inherit; text-decoration: inherit; }
            div.sourceCode { margin: 1em 0; }
            pre.sourceCode { margin: 0; }
            @media screen {
            div.sourceCode { overflow: auto; }
            }
            @media print {
            pre > code.sourceCode { white-space: pre-wrap; }
            pre > code.sourceCode > span { text-indent: -5em; padding-left: 5em; }
            }
            pre.numberSource code
              { counter-reset: source-line 0; }
            pre.numberSource code > span
              { position: relative; left: -4em; counter-increment: source-line; }
            pre.numberSource code > span > a:first-child::before
              { content: counter(source-line);
                position: relative; left: -1em; text-align: right; vertical-align: baseline;
                border: none; display: inline-block;
                -webkit-touch-callout: none; -webkit-user-select: none;
                -khtml-user-select: none; -moz-user-select: none;
                -ms-user-select: none; user-select: none;
                padding: 0 4px; width: 4em;
                color: #aaaaaa;
              }
            pre.numberSource { margin-left: 3em; border-left: 1px solid #aaaaaa;  padding-left: 4px; }
            div.sourceCode
              {   }
            @media screen {
            pre > code.sourceCode > span > a:first-child::before { text-decoration: underline; }
            }
            code span.al { color: #ff0000; font-weight: bold; } /* Alert */
            code span.an { color: #60a0b0; font-weight: bold; font-style: italic; } /* Annotation */
            code span.at { color: #7d9029; } /* Attribute */
            code span.bn { color: #40a070; } /* BaseN */
            code span.bu { } /* BuiltIn */
            code span.cf { color: #007020; font-weight: bold; } /* ControlFlow */
            code span.ch { color: #4070a0; } /* Char */
            code span.cn { color: #880000; } /* Constant */
            code span.co { color: #60a0b0; font-style: italic; } /* Comment */
            code span.cv { color: #60a0b0; font-weight: bold; font-style: italic; } /* CommentVar */
            code span.do { color: #ba2121; font-style: italic; } /* Documentation */
            code span.dt { color: #902000; } /* DataType */
            code span.dv { color: #40a070; } /* DecVal */
            code span.er { color: #ff0000; font-weight: bold; } /* Error */
            code span.ex { } /* Extension */
            code span.fl { color: #40a070; } /* Float */
            code span.fu { color: #06287e; } /* Function */
            code span.im { } /* Import */
            code span.in { color: #60a0b0; font-weight: bold; font-style: italic; } /* Information */
            code span.kw { color: #007020; font-weight: bold; } /* Keyword */
            code span.op { color: #666666; } /* Operator */
            code span.ot { color: #007020; } /* Other */
            code span.pp { color: #bc7a00; } /* Preprocessor */
            code span.sc { color: #4070a0; } /* SpecialChar */
            code span.ss { color: #bb6688; } /* SpecialString */
            code span.st { color: #4070a0; } /* String */
            code span.va { color: #19177c; } /* Variable */
            code span.vs { color: #4070a0; } /* VerbatimString */
            code span.wa { color: #60a0b0; font-weight: bold; font-style: italic; } /* Warning */
            .display.math{display: block; text-align: center; margin: 0.5rem auto;}
          </style>
          <!--[if lt IE 9]>
            <script src="//cdnjs.cloudflare.com/ajax/libs/html5shiv/3.7.3/html5shiv-printshiv.min.js"></script>
          <![endif]-->
    <pre><code>&lt;/head&gt;
    &lt;body&gt;
    &lt;h1&gt;
    Table of Contents&lt;span class=&quot;tocSkip&quot;&gt;&lt;/span&gt;
    &lt;/h1&gt;
    &lt;div class=&quot;toc&quot;&gt;
    &lt;ul class=&quot;toc-item&quot;&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Utility-Scripts&quot; data-toc-modified-id=&quot;Utility-Scripts-1&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;1  &lt;/span&gt;Utility Scripts&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Modeling-rationale&quot; data-toc-modified-id=&quot;Modeling-rationale-2&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;2  &lt;/span&gt;Modeling rationale&lt;/a&gt;&lt;/span&gt;
    &lt;ul class=&quot;toc-item&quot;&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#The-internal-clock&quot; data-toc-modified-id=&quot;The-internal-clock-2.1&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;2.1  &lt;/span&gt;The internal clock&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Synchronization-process&quot; data-toc-modified-id=&quot;Synchronization-process-2.2&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;2.2  &lt;/span&gt;Synchronization process&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;/ul&gt;
    &lt;/li&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Cellular-Automata-modeling&quot; data-toc-modified-id=&quot;Cellular-Automata-modeling-3&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;3  &lt;/span&gt;Cellular Automata modeling&lt;/a&gt;&lt;/span&gt;
    &lt;ul class=&quot;toc-item&quot;&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Parameters-of-the-simulation&quot; data-toc-modified-id=&quot;Parameters-of-the-simulation-3.1&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;3.1  &lt;/span&gt;Parameters of the simulation&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Simulation-script&quot; data-toc-modified-id=&quot;Simulation-script-3.2&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;3.2  &lt;/span&gt;Simulation script&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;li&gt;
    &lt;span&gt;&lt;a href=&quot;#Simulation&quot; data-toc-modified-id=&quot;Simulation-3.3&quot;&gt;&lt;span class=&quot;toc-item-num&quot;&gt;3.3  &lt;/span&gt;Simulation&lt;/a&gt;&lt;/span&gt;
    &lt;/li&gt;
    &lt;/ul&gt;
    &lt;/li&gt;
    &lt;/ul&gt;
    &lt;/div&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb1&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb1-1&quot;&gt;&lt;a href=&quot;#cb1-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# before everything, import these libraries (Shift-Enter)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb1-2&quot;&gt;&lt;a href=&quot;#cb1-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; os&lt;/span&gt;
    &lt;span id=&quot;cb1-3&quot;&gt;&lt;a href=&quot;#cb1-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; numpy &lt;span class=&quot;im&quot;&gt;as&lt;/span&gt; np&lt;/span&gt;
    &lt;span id=&quot;cb1-4&quot;&gt;&lt;a href=&quot;#cb1-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; xarray &lt;span class=&quot;im&quot;&gt;as&lt;/span&gt; xr&lt;/span&gt;
    &lt;span id=&quot;cb1-5&quot;&gt;&lt;a href=&quot;#cb1-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; plotly.graph_objects &lt;span class=&quot;im&quot;&gt;as&lt;/span&gt; go&lt;/span&gt;
    &lt;span id=&quot;cb1-6&quot;&gt;&lt;a href=&quot;#cb1-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; ipywidgets &lt;span class=&quot;im&quot;&gt;as&lt;/span&gt; widgets&lt;/span&gt;
    &lt;span id=&quot;cb1-7&quot;&gt;&lt;a href=&quot;#cb1-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;from&lt;/span&gt; fractions &lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; Fraction&lt;/span&gt;
    &lt;span id=&quot;cb1-8&quot;&gt;&lt;a href=&quot;#cb1-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;from&lt;/span&gt; dataclasses &lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; dataclass&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;h1 id=&quot;fireflies-synchronization&quot;&gt;Fireflies synchronization&lt;/h1&gt;
    &lt;hr&gt;
    &lt;p&gt; &lt;/p&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb2&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb2-1&quot;&gt;&lt;a href=&quot;#cb2-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;im&quot;&gt;from&lt;/span&gt; IPython.lib.display &lt;span class=&quot;im&quot;&gt;import&lt;/span&gt; YouTubeVideo&lt;/span&gt;
    &lt;span id=&quot;cb2-2&quot;&gt;&lt;a href=&quot;#cb2-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;YouTubeVideo(&lt;span class=&quot;st&quot;&gt;&amp;#39;https://www.youtube.com/watch?v=d77GdblhvEo&amp;#39;&lt;/span&gt;,width&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;560&lt;/span&gt;,height&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;315&lt;/span&gt;)&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;iframe width=&quot;560&quot; height=&quot;315&quot; src=&quot;https://www.youtube.com/embed/https://www.youtube.com/watch?v=d77GdblhvEo&quot; frameborder=&quot;0&quot; allowfullscreen&gt;
    &lt;/iframe&gt;
    &lt;h2 id=&quot;utility-scripts&quot;&gt;Utility Scripts&lt;/h2&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb3&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb3-1&quot;&gt;&lt;a href=&quot;#cb3-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;at&quot;&gt;@dataclass&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-2&quot;&gt;&lt;a href=&quot;#cb3-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;class&lt;/span&gt; Slider:&lt;/span&gt;
    &lt;span id=&quot;cb3-3&quot;&gt;&lt;a href=&quot;#cb3-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;Represent a range of (linear) values as both:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-4&quot;&gt;&lt;a href=&quot;#cb3-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    - an np.array &lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-5&quot;&gt;&lt;a href=&quot;#cb3-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    - an ipywidget.Floatslider&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-6&quot;&gt;&lt;a href=&quot;#cb3-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-7&quot;&gt;&lt;a href=&quot;#cb3-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    name: &lt;span class=&quot;bu&quot;&gt;str&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-8&quot;&gt;&lt;a href=&quot;#cb3-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    start: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-9&quot;&gt;&lt;a href=&quot;#cb3-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    stop: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-10&quot;&gt;&lt;a href=&quot;#cb3-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    step: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-11&quot;&gt;&lt;a href=&quot;#cb3-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    val_ini: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;None&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-12&quot;&gt;&lt;a href=&quot;#cb3-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-13&quot;&gt;&lt;a href=&quot;#cb3-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; __post_init__(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;):&lt;/span&gt;
    &lt;span id=&quot;cb3-14&quot;&gt;&lt;a href=&quot;#cb3-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; nparange(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start, &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop, &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.step)&lt;/span&gt;
    &lt;span id=&quot;cb3-15&quot;&gt;&lt;a href=&quot;#cb3-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; &lt;span class=&quot;kw&quot;&gt;not&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini:&lt;/span&gt;
    &lt;span id=&quot;cb3-16&quot;&gt;&lt;a href=&quot;#cb3-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.random.choice(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val, &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;]&lt;/span&gt;
    &lt;span id=&quot;cb3-17&quot;&gt;&lt;a href=&quot;#cb3-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;/span&gt;
    &lt;span id=&quot;cb3-18&quot;&gt;&lt;a href=&quot;#cb3-18&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.slider &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; widgets.FloatSlider(&lt;span class=&quot;bu&quot;&gt;min&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start,&lt;/span&gt;
    &lt;span id=&quot;cb3-19&quot;&gt;&lt;a href=&quot;#cb3-19&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        &lt;span class=&quot;bu&quot;&gt;max&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop,&lt;/span&gt;
    &lt;span id=&quot;cb3-20&quot;&gt;&lt;a href=&quot;#cb3-20&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.step,&lt;/span&gt;
    &lt;span id=&quot;cb3-21&quot;&gt;&lt;a href=&quot;#cb3-21&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        value&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini,&lt;/span&gt;
    &lt;span id=&quot;cb3-22&quot;&gt;&lt;a href=&quot;#cb3-22&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        description&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.name,&lt;/span&gt;
    &lt;span id=&quot;cb3-23&quot;&gt;&lt;a href=&quot;#cb3-23&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        continuous_update&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;True&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb3-24&quot;&gt;&lt;a href=&quot;#cb3-24&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-25&quot;&gt;&lt;a href=&quot;#cb3-25&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-26&quot;&gt;&lt;a href=&quot;#cb3-26&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;at&quot;&gt;@dataclass&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-27&quot;&gt;&lt;a href=&quot;#cb3-27&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;class&lt;/span&gt; logSlider:&lt;/span&gt;
    &lt;span id=&quot;cb3-28&quot;&gt;&lt;a href=&quot;#cb3-28&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;Represent a range of log values as both:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-29&quot;&gt;&lt;a href=&quot;#cb3-29&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    - an np.array &lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-30&quot;&gt;&lt;a href=&quot;#cb3-30&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    - an ipywidget.FloatLogSlider&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-31&quot;&gt;&lt;a href=&quot;#cb3-31&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-32&quot;&gt;&lt;a href=&quot;#cb3-32&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    name: &lt;span class=&quot;bu&quot;&gt;str&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-33&quot;&gt;&lt;a href=&quot;#cb3-33&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    start: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-34&quot;&gt;&lt;a href=&quot;#cb3-34&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    stop: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-35&quot;&gt;&lt;a href=&quot;#cb3-35&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    num: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-36&quot;&gt;&lt;a href=&quot;#cb3-36&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    val_ini: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;None&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-37&quot;&gt;&lt;a href=&quot;#cb3-37&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    base: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-38&quot;&gt;&lt;a href=&quot;#cb3-38&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    decimals: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-39&quot;&gt;&lt;a href=&quot;#cb3-39&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-40&quot;&gt;&lt;a href=&quot;#cb3-40&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; __post_init__(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;):&lt;/span&gt;
    &lt;span id=&quot;cb3-41&quot;&gt;&lt;a href=&quot;#cb3-41&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# create numpy array of all values&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-42&quot;&gt;&lt;a href=&quot;#cb3-42&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.around(np.logspace(start&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start,&lt;/span&gt;
    &lt;span id=&quot;cb3-43&quot;&gt;&lt;a href=&quot;#cb3-43&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        stop&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop,&lt;/span&gt;
    &lt;span id=&quot;cb3-44&quot;&gt;&lt;a href=&quot;#cb3-44&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        num&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.num, &lt;/span&gt;
    &lt;span id=&quot;cb3-45&quot;&gt;&lt;a href=&quot;#cb3-45&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                        endpoint&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;True&lt;/span&gt;), &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.decimals)&lt;/span&gt;
    &lt;span id=&quot;cb3-46&quot;&gt;&lt;a href=&quot;#cb3-46&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# check each value is unique&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-47&quot;&gt;&lt;a href=&quot;#cb3-47&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val.size &lt;span class=&quot;op&quot;&gt;!=&lt;/span&gt; np.unique(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val, return_counts&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;False&lt;/span&gt;).size:&lt;/span&gt;
    &lt;span id=&quot;cb3-48&quot;&gt;&lt;a href=&quot;#cb3-48&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;bu&quot;&gt;print&lt;/span&gt;(&lt;span class=&quot;ss&quot;&gt;f&amp;quot;WARNING: Repeated values in &lt;/span&gt;&lt;span class=&quot;sc&quot;&gt;{&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;&lt;span class=&quot;sc&quot;&gt;.&lt;/span&gt;name&lt;span class=&quot;sc&quot;&gt;}&lt;/span&gt;&lt;span class=&quot;ss&quot;&gt;.val&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-49&quot;&gt;&lt;a href=&quot;#cb3-49&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                &lt;span class=&quot;st&quot;&gt;&amp;quot;, increase &amp;#39;decimals&amp;#39; or reduce &amp;#39;num&amp;#39;&amp;quot;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb3-50&quot;&gt;&lt;a href=&quot;#cb3-50&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-51&quot;&gt;&lt;a href=&quot;#cb3-51&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# pick initial value if not provided&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-52&quot;&gt;&lt;a href=&quot;#cb3-52&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; &lt;span class=&quot;kw&quot;&gt;not&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini:&lt;/span&gt;
    &lt;span id=&quot;cb3-53&quot;&gt;&lt;a href=&quot;#cb3-53&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.random.choice(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val, &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;]&lt;/span&gt;
    &lt;span id=&quot;cb3-54&quot;&gt;&lt;a href=&quot;#cb3-54&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-55&quot;&gt;&lt;a href=&quot;#cb3-55&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# convert num into step for FloatLogSlider&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-56&quot;&gt;&lt;a href=&quot;#cb3-56&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        step &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; (&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop &lt;span class=&quot;op&quot;&gt;-&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start)&lt;span class=&quot;op&quot;&gt;/&lt;/span&gt;(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.num&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb3-57&quot;&gt;&lt;a href=&quot;#cb3-57&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-58&quot;&gt;&lt;a href=&quot;#cb3-58&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# create slider&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-59&quot;&gt;&lt;a href=&quot;#cb3-59&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.slider &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; widgets.FloatLogSlider(&lt;span class=&quot;bu&quot;&gt;min&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start,&lt;/span&gt;
    &lt;span id=&quot;cb3-60&quot;&gt;&lt;a href=&quot;#cb3-60&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    &lt;span class=&quot;bu&quot;&gt;max&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop,&lt;/span&gt;
    &lt;span id=&quot;cb3-61&quot;&gt;&lt;a href=&quot;#cb3-61&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;step,&lt;/span&gt;
    &lt;span id=&quot;cb3-62&quot;&gt;&lt;a href=&quot;#cb3-62&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    value&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini,&lt;/span&gt;
    &lt;span id=&quot;cb3-63&quot;&gt;&lt;a href=&quot;#cb3-63&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    base&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.base,&lt;/span&gt;
    &lt;span id=&quot;cb3-64&quot;&gt;&lt;a href=&quot;#cb3-64&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    description&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.name,&lt;/span&gt;
    &lt;span id=&quot;cb3-65&quot;&gt;&lt;a href=&quot;#cb3-65&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                    readout_format&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;ss&quot;&gt;f&amp;#39;.&lt;/span&gt;&lt;span class=&quot;sc&quot;&gt;{&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;&lt;span class=&quot;sc&quot;&gt;.&lt;/span&gt;decimals&lt;span class=&quot;sc&quot;&gt;}&lt;/span&gt;&lt;span class=&quot;ss&quot;&gt;f&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb3-66&quot;&gt;&lt;a href=&quot;#cb3-66&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-67&quot;&gt;&lt;a href=&quot;#cb3-67&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-68&quot;&gt;&lt;a href=&quot;#cb3-68&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;at&quot;&gt;@dataclass&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-69&quot;&gt;&lt;a href=&quot;#cb3-69&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;class&lt;/span&gt; AnimateSlider:&lt;/span&gt;
    &lt;span id=&quot;cb3-70&quot;&gt;&lt;a href=&quot;#cb3-70&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    start: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-71&quot;&gt;&lt;a href=&quot;#cb3-71&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    stop: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-72&quot;&gt;&lt;a href=&quot;#cb3-72&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    step: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-73&quot;&gt;&lt;a href=&quot;#cb3-73&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    name: &lt;span class=&quot;bu&quot;&gt;str&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;st&quot;&gt;&amp;quot;Press play&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-74&quot;&gt;&lt;a href=&quot;#cb3-74&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    val_ini: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;None&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-75&quot;&gt;&lt;a href=&quot;#cb3-75&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    interval: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;100&lt;/span&gt;  &lt;span class=&quot;co&quot;&gt;# time interval (in ms)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-76&quot;&gt;&lt;a href=&quot;#cb3-76&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-77&quot;&gt;&lt;a href=&quot;#cb3-77&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; __post_init__(&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;):&lt;/span&gt;
    &lt;span id=&quot;cb3-78&quot;&gt;&lt;a href=&quot;#cb3-78&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# create the play widget&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-79&quot;&gt;&lt;a href=&quot;#cb3-79&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.play &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; widgets.Play(&lt;/span&gt;
    &lt;span id=&quot;cb3-80&quot;&gt;&lt;a href=&quot;#cb3-80&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;bu&quot;&gt;min&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start,&lt;/span&gt;
    &lt;span id=&quot;cb3-81&quot;&gt;&lt;a href=&quot;#cb3-81&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;bu&quot;&gt;max&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop,&lt;/span&gt;
    &lt;span id=&quot;cb3-82&quot;&gt;&lt;a href=&quot;#cb3-82&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            step &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.step,&lt;/span&gt;
    &lt;span id=&quot;cb3-83&quot;&gt;&lt;a href=&quot;#cb3-83&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            interval &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.interval,&lt;/span&gt;
    &lt;span id=&quot;cb3-84&quot;&gt;&lt;a href=&quot;#cb3-84&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            description &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.name)&lt;/span&gt;
    &lt;span id=&quot;cb3-85&quot;&gt;&lt;a href=&quot;#cb3-85&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-86&quot;&gt;&lt;a href=&quot;#cb3-86&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini:&lt;/span&gt;
    &lt;span id=&quot;cb3-87&quot;&gt;&lt;a href=&quot;#cb3-87&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.play.value &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.val_ini&lt;/span&gt;
    &lt;span id=&quot;cb3-88&quot;&gt;&lt;a href=&quot;#cb3-88&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-89&quot;&gt;&lt;a href=&quot;#cb3-89&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# create a slider for visualization&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-90&quot;&gt;&lt;a href=&quot;#cb3-90&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.slider &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; widgets.IntSlider(&lt;/span&gt;
    &lt;span id=&quot;cb3-91&quot;&gt;&lt;a href=&quot;#cb3-91&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                &lt;span class=&quot;bu&quot;&gt;min&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.start,&lt;/span&gt;
    &lt;span id=&quot;cb3-92&quot;&gt;&lt;a href=&quot;#cb3-92&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                &lt;span class=&quot;bu&quot;&gt;max&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.stop,&lt;/span&gt;
    &lt;span id=&quot;cb3-93&quot;&gt;&lt;a href=&quot;#cb3-93&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                step &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.step)&lt;/span&gt;
    &lt;span id=&quot;cb3-94&quot;&gt;&lt;a href=&quot;#cb3-94&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-95&quot;&gt;&lt;a href=&quot;#cb3-95&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;co&quot;&gt;# Link the slider and the play widget&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-96&quot;&gt;&lt;a href=&quot;#cb3-96&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        widgets.jslink((&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.play, &lt;span class=&quot;st&quot;&gt;&amp;#39;value&amp;#39;&lt;/span&gt;), (&lt;span class=&quot;va&quot;&gt;self&lt;/span&gt;.slider, &lt;span class=&quot;st&quot;&gt;&amp;#39;value&amp;#39;&lt;/span&gt;))&lt;/span&gt;
    &lt;span id=&quot;cb3-97&quot;&gt;&lt;a href=&quot;#cb3-97&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-98&quot;&gt;&lt;a href=&quot;#cb3-98&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-99&quot;&gt;&lt;a href=&quot;#cb3-99&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; nparange(start, stop, step):&lt;/span&gt;
    &lt;span id=&quot;cb3-100&quot;&gt;&lt;a href=&quot;#cb3-100&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;Modified np.arange()&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-101&quot;&gt;&lt;a href=&quot;#cb3-101&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        - improve float precision (by use of fractions)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-102&quot;&gt;&lt;a href=&quot;#cb3-102&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        - includes endpoint&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-103&quot;&gt;&lt;a href=&quot;#cb3-103&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-104&quot;&gt;&lt;a href=&quot;#cb3-104&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Args:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-105&quot;&gt;&lt;a href=&quot;#cb3-105&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        start, stop, step: float (stop is included in array)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-106&quot;&gt;&lt;a href=&quot;#cb3-106&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-107&quot;&gt;&lt;a href=&quot;#cb3-107&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Returns:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-108&quot;&gt;&lt;a href=&quot;#cb3-108&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        ndarray&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-109&quot;&gt;&lt;a href=&quot;#cb3-109&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-110&quot;&gt;&lt;a href=&quot;#cb3-110&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    delta, zoom  &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; get_frac(step)&lt;/span&gt;
    &lt;span id=&quot;cb3-111&quot;&gt;&lt;a href=&quot;#cb3-111&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;/span&gt;
    &lt;span id=&quot;cb3-112&quot;&gt;&lt;a href=&quot;#cb3-112&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;return&lt;/span&gt; np.arange(start &lt;span class=&quot;op&quot;&gt;*&lt;/span&gt; zoom, stop &lt;span class=&quot;op&quot;&gt;*&lt;/span&gt; zoom &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; delta, delta) &lt;span class=&quot;op&quot;&gt;/&lt;/span&gt; zoom&lt;/span&gt;
    &lt;span id=&quot;cb3-113&quot;&gt;&lt;a href=&quot;#cb3-113&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-114&quot;&gt;&lt;a href=&quot;#cb3-114&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-115&quot;&gt;&lt;a href=&quot;#cb3-115&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; get_frac(step, readout_format&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;.16f&amp;#39;&lt;/span&gt;, atol&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;1e-12&lt;/span&gt;):&lt;/span&gt;
    &lt;span id=&quot;cb3-116&quot;&gt;&lt;a href=&quot;#cb3-116&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    precision &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;st&quot;&gt;&amp;quot;{:&amp;quot;&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; readout_format &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; &lt;span class=&quot;st&quot;&gt;&amp;quot;}&amp;quot;&lt;/span&gt; &lt;/span&gt;
    &lt;span id=&quot;cb3-117&quot;&gt;&lt;a href=&quot;#cb3-117&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    frac &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; Fraction(precision.&lt;span class=&quot;bu&quot;&gt;format&lt;/span&gt;(step))&lt;/span&gt;
    &lt;span id=&quot;cb3-118&quot;&gt;&lt;a href=&quot;#cb3-118&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; frac.denominator &lt;span class=&quot;op&quot;&gt;&amp;gt;&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;/&lt;/span&gt;atol:&lt;/span&gt;
    &lt;span id=&quot;cb3-119&quot;&gt;&lt;a href=&quot;#cb3-119&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;bu&quot;&gt;print&lt;/span&gt;(&lt;span class=&quot;st&quot;&gt;&amp;quot;WARNING: potential Floats inconsistencies due to &amp;#39;step&amp;#39;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb3-120&quot;&gt;&lt;a href=&quot;#cb3-120&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;st&quot;&gt;&amp;quot; being an irrational number&amp;quot;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb3-121&quot;&gt;&lt;a href=&quot;#cb3-121&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;return&lt;/span&gt; (frac.numerator, frac.denominator)&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;h2 id=&quot;modeling-rationale&quot;&gt;Modeling rationale&lt;/h2&gt;
    &lt;h3 id=&quot;the-internal-clock&quot;&gt;The internal clock&lt;/h3&gt;
    &lt;ul&gt;
    &lt;li&gt;Each firefly has its own individual internal clock (or phase) &lt;span class=&quot;math inline&quot;&gt;&lt;em&gt;θ&lt;/em&gt;&lt;/span&gt;&lt;/li&gt;
    &lt;li&gt;&lt;span class=&quot;math inline&quot;&gt;&lt;em&gt;θ&lt;/em&gt;&lt;/span&gt; has a period T&lt;/li&gt;
    &lt;li&gt;&lt;span class=&quot;math inline&quot;&gt;&lt;em&gt;θ&lt;/em&gt;&lt;/span&gt; varies between 0 and 1&lt;/li&gt;
    &lt;li&gt;Every time the clock reachs 1 (every T times), the firefly flashes&lt;/li&gt;
    &lt;li&gt;After flashing, the clock is reset to 0&lt;/li&gt;
    &lt;/ul&gt;
    &lt;p&gt;&lt;span class=&quot;math display&quot;&gt;$$\begin{align*}
    &amp;amp;\theta_{t+1} =  \theta_t + \frac{1}{T} &amp;amp;&amp;amp; \text{ (mod 1)} \\
    \Rightarrow&amp;amp; \theta_{t+1} - \theta_t = \frac{1}{T} &amp;amp;&amp;amp;  \text{ (mod 1)} \\
    \Rightarrow&amp;amp; \frac{\theta_{t+1} - \theta_t}{(t+1) - t} = \frac{1}{T} &amp;amp;&amp;amp; \text{ (mod 1)} \\
    \Rightarrow&amp;amp; \theta(t) = \frac{t}{T} &amp;amp;&amp;amp; \text{ (mod 1)}
    \end{align*}$$&lt;/span&gt;&lt;/p&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb4&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb4-1&quot;&gt;&lt;a href=&quot;#cb4-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Let&amp;#39;s visualize what we just said&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-2&quot;&gt;&lt;a href=&quot;#cb4-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-3&quot;&gt;&lt;a href=&quot;#cb4-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# set the variable&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-4&quot;&gt;&lt;a href=&quot;#cb4-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;t1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.linspace(&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;, &lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;, &lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;)  &lt;span class=&quot;co&quot;&gt;# time array&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-5&quot;&gt;&lt;a href=&quot;#cb4-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-6&quot;&gt;&lt;a href=&quot;#cb4-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set a slider to check the influence of the resetting strength&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-7&quot;&gt;&lt;a href=&quot;#cb4-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;T1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; Slider(name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;period T&amp;#39;&lt;/span&gt;, start&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;, stop&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;, step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb4-8&quot;&gt;&lt;a href=&quot;#cb4-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-9&quot;&gt;&lt;a href=&quot;#cb4-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set an xarray for all (time, amplitude) value combinations&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-10&quot;&gt;&lt;a href=&quot;#cb4-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;f1&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;kw&quot;&gt;lambda&lt;/span&gt; t, T: np.mod(t&lt;span class=&quot;op&quot;&gt;/&lt;/span&gt;T, &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb4-11&quot;&gt;&lt;a href=&quot;#cb4-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;tt1, TT &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.meshgrid(t1, T1.val)&lt;/span&gt;
    &lt;span id=&quot;cb4-12&quot;&gt;&lt;a href=&quot;#cb4-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;y1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; xr.DataArray(f1(tt1, TT),&lt;/span&gt;
    &lt;span id=&quot;cb4-13&quot;&gt;&lt;a href=&quot;#cb4-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            dims&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;[&lt;span class=&quot;st&quot;&gt;&amp;#39;T&amp;#39;&lt;/span&gt;, &lt;span class=&quot;st&quot;&gt;&amp;#39;t&amp;#39;&lt;/span&gt;],&lt;/span&gt;
    &lt;span id=&quot;cb4-14&quot;&gt;&lt;a href=&quot;#cb4-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            coords&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;{&lt;span class=&quot;st&quot;&gt;&amp;#39;T&amp;#39;&lt;/span&gt;: T1.val, &lt;span class=&quot;st&quot;&gt;&amp;#39;t&amp;#39;&lt;/span&gt;: t1})&lt;/span&gt;
    &lt;span id=&quot;cb4-15&quot;&gt;&lt;a href=&quot;#cb4-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-16&quot;&gt;&lt;a href=&quot;#cb4-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-17&quot;&gt;&lt;a href=&quot;#cb4-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the graph&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-18&quot;&gt;&lt;a href=&quot;#cb4-18&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;trace0 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.Scatter(x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;t1, &lt;/span&gt;
    &lt;span id=&quot;cb4-19&quot;&gt;&lt;a href=&quot;#cb4-19&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    y&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;y1.sel(T&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;T1.val_ini).values)&lt;/span&gt;
    &lt;span id=&quot;cb4-20&quot;&gt;&lt;a href=&quot;#cb4-20&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.FigureWidget([trace0])&lt;/span&gt;
    &lt;span id=&quot;cb4-21&quot;&gt;&lt;a href=&quot;#cb4-21&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig1.update_layout(template&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;none&amp;#39;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb4-22&quot;&gt;&lt;a href=&quot;#cb4-22&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    width&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;800&lt;/span&gt;, height&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;500&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb4-23&quot;&gt;&lt;a href=&quot;#cb4-23&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;Flashes of a single firefly&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb4-24&quot;&gt;&lt;a href=&quot;#cb4-24&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title_x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.5&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb4-25&quot;&gt;&lt;a href=&quot;#cb4-25&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    xaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;t&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb4-26&quot;&gt;&lt;a href=&quot;#cb4-26&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    yaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;θ&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb4-27&quot;&gt;&lt;a href=&quot;#cb4-27&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-28&quot;&gt;&lt;a href=&quot;#cb4-28&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-29&quot;&gt;&lt;a href=&quot;#cb4-29&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the callback to update the graph &lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-30&quot;&gt;&lt;a href=&quot;#cb4-30&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; update1(change):&lt;/span&gt;
    &lt;span id=&quot;cb4-31&quot;&gt;&lt;a href=&quot;#cb4-31&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    fig1.data[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].y &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; y1.sel(T&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;change[&lt;span class=&quot;st&quot;&gt;&amp;#39;new&amp;#39;&lt;/span&gt;]).values&lt;/span&gt;
    &lt;span id=&quot;cb4-32&quot;&gt;&lt;a href=&quot;#cb4-32&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-33&quot;&gt;&lt;a href=&quot;#cb4-33&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-34&quot;&gt;&lt;a href=&quot;#cb4-34&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Link the slider and the callback&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-35&quot;&gt;&lt;a href=&quot;#cb4-35&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;T1.slider.observe(update1, names&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;value&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb4-36&quot;&gt;&lt;a href=&quot;#cb4-36&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-37&quot;&gt;&lt;a href=&quot;#cb4-37&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Display&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb4-38&quot;&gt;&lt;a href=&quot;#cb4-38&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;display(widgets.VBox([T1.slider, fig1]))&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;pre&gt;&lt;code&gt;VBox(children=(FloatSlider(value=2.0, description=&amp;#39;period T&amp;#39;, max=10.0, min=1.0, step=1.0), FigureWidget({
       …&lt;/code&gt;&lt;/pre&gt;
    &lt;h3 id=&quot;synchronization-process&quot;&gt;Synchronization process&lt;/h3&gt;
    &lt;ul&gt;
    &lt;li&gt;When a firefly flashes, it influences its neighbours&lt;/li&gt;
    &lt;li&gt;The neighbours slows down or speeds up so as to flash more nearly in phase on the next cycle&lt;/li&gt;
    &lt;li&gt;A simple model satisfying this hypothesis is:&lt;/li&gt;
    &lt;/ul&gt;
    &lt;p&gt;&lt;/p&gt;
    &lt;p&gt;where &lt;span class=&quot;math inline&quot;&gt;&lt;em&gt;Θ&lt;/em&gt;&lt;sub&gt;&lt;em&gt;t&lt;/em&gt;&lt;/sub&gt;&lt;/span&gt; is the phase of a flashing (neigbhoring) firely[^1].&lt;/p&gt;
    &lt;p&gt;&lt;/p&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb6&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb6-1&quot;&gt;&lt;a href=&quot;#cb6-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# set the variable&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-2&quot;&gt;&lt;a href=&quot;#cb6-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;t2 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.linspace(&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;,&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;,&lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;)  &lt;span class=&quot;co&quot;&gt;# time array&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-3&quot;&gt;&lt;a href=&quot;#cb6-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;T2 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;200&lt;/span&gt;                    &lt;span class=&quot;co&quot;&gt;# 

    [WARNING] Could not convert TeX math \begin{align*}
      &\theta_{t+1} =  \theta_t + \frac{1}{T} && \text{ (mod 1)} \\
      \Rightarrow& \theta_{t+1} - \theta_t = \frac{1}{T} &&  \text{ (mod 1)} \\
      \Rightarrow& \frac{\theta_{t+1} - \theta_t}{(t+1) - t} = \frac{1}{T} && \text{ (mod 1)} \\
      \Rightarrow& \theta(t) = \frac{t}{T} && \text{ (mod 1)}
      \end{align*}, rendering as TeX
    [WARNING] Could not convert TeX math \begin{align*}\theta_{t+1} =  \theta_t + \frac{1}{T} + A\sin(\Theta_t - \theta_t)\end{align*}, rendering as TeX
    [WARNING] This document format requires a nonempty <title> element.
      Defaulting to 'Fireflies-Synchronization' as the title.
      To specify a title, use 'title' in metadata or --metadata title="...".





    0



    the period&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-4&quot;&gt;&lt;a href=&quot;#cb6-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-5&quot;&gt;&lt;a href=&quot;#cb6-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-6&quot;&gt;&lt;a href=&quot;#cb6-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set a slider to check the influence of the resetting strength&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-7&quot;&gt;&lt;a href=&quot;#cb6-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;A &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; Slider(name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;Amplitude&amp;#39;&lt;/span&gt;, start&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;, stop&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;, step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.01&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-8&quot;&gt;&lt;a href=&quot;#cb6-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;A.slider.description_tooltip &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;st&quot;&gt;&amp;quot;Amplitude of the resetting strength&lt;/span&gt;&lt;span class=&quot;ch&quot;&gt;\n&lt;/span&gt;&lt;span class=&quot;st&quot;&gt; &amp;quot;&lt;/span&gt;\&lt;/span&gt;
    &lt;span id=&quot;cb6-9&quot;&gt;&lt;a href=&quot;#cb6-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                &lt;span class=&quot;st&quot;&gt;&amp;quot;(it measures the firefly’s ability to &amp;quot;&lt;/span&gt;\&lt;/span&gt;
    &lt;span id=&quot;cb6-10&quot;&gt;&lt;a href=&quot;#cb6-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                &lt;span class=&quot;st&quot;&gt;&amp;quot;modify its instantaneous frequency)&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-11&quot;&gt;&lt;a href=&quot;#cb6-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-12&quot;&gt;&lt;a href=&quot;#cb6-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set an xarray for all (time, amplitude) value combinations&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-13&quot;&gt;&lt;a href=&quot;#cb6-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Option 1: using modulus&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-14&quot;&gt;&lt;a href=&quot;#cb6-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;f2&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;kw&quot;&gt;lambda&lt;/span&gt; t, A: np.mod(t &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;/&lt;/span&gt;T2 &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; A&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;np.sin(&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;np.pi&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;(&lt;span class=&quot;fl&quot;&gt;1.005&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;t)), &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-15&quot;&gt;&lt;a href=&quot;#cb6-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;tt2, AA &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.meshgrid(t2, A.val)&lt;/span&gt;
    &lt;span id=&quot;cb6-16&quot;&gt;&lt;a href=&quot;#cb6-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;y2 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; xr.DataArray(f2(tt2, AA),&lt;/span&gt;
    &lt;span id=&quot;cb6-17&quot;&gt;&lt;a href=&quot;#cb6-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            dims&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;[&lt;span class=&quot;st&quot;&gt;&amp;#39;Amplitude&amp;#39;&lt;/span&gt;, &lt;span class=&quot;st&quot;&gt;&amp;#39;t&amp;#39;&lt;/span&gt;],&lt;/span&gt;
    &lt;span id=&quot;cb6-18&quot;&gt;&lt;a href=&quot;#cb6-18&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            coords&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;{&lt;span class=&quot;st&quot;&gt;&amp;#39;Amplitude&amp;#39;&lt;/span&gt;: A.val, &lt;span class=&quot;st&quot;&gt;&amp;#39;t&amp;#39;&lt;/span&gt;: t2})&lt;/span&gt;
    &lt;span id=&quot;cb6-19&quot;&gt;&lt;a href=&quot;#cb6-19&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-20&quot;&gt;&lt;a href=&quot;#cb6-20&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-21&quot;&gt;&lt;a href=&quot;#cb6-21&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-22&quot;&gt;&lt;a href=&quot;#cb6-22&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the graph&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-23&quot;&gt;&lt;a href=&quot;#cb6-23&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;trace0 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.Scatter(x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;t2, &lt;/span&gt;
    &lt;span id=&quot;cb6-24&quot;&gt;&lt;a href=&quot;#cb6-24&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    y&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;y2.sel(Amplitude&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;A.val_ini).values,&lt;/span&gt;
    &lt;span id=&quot;cb6-25&quot;&gt;&lt;a href=&quot;#cb6-25&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;with coupling&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-26&quot;&gt;&lt;a href=&quot;#cb6-26&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;trace1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.Scatter(x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;t2,&lt;/span&gt;
    &lt;span id=&quot;cb6-27&quot;&gt;&lt;a href=&quot;#cb6-27&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    y&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;np.mod(t2&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;(&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;/&lt;/span&gt;T2),&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;),&lt;/span&gt;
    &lt;span id=&quot;cb6-28&quot;&gt;&lt;a href=&quot;#cb6-28&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;no coupling&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-29&quot;&gt;&lt;a href=&quot;#cb6-29&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig2 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.FigureWidget([trace0, trace1])&lt;/span&gt;
    &lt;span id=&quot;cb6-30&quot;&gt;&lt;a href=&quot;#cb6-30&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig2.update_layout(template&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;none&amp;#39;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-31&quot;&gt;&lt;a href=&quot;#cb6-31&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    width&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;800&lt;/span&gt;, height&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;500&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-32&quot;&gt;&lt;a href=&quot;#cb6-32&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;Influence of the resetting strength&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-33&quot;&gt;&lt;a href=&quot;#cb6-33&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title_x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.5&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-34&quot;&gt;&lt;a href=&quot;#cb6-34&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    xaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;θ&amp;lt;sub&amp;gt;t&amp;lt;sub&amp;gt;&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-35&quot;&gt;&lt;a href=&quot;#cb6-35&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    yaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;θ&amp;lt;sub&amp;gt;t+1&amp;lt;sub&amp;gt;&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb6-36&quot;&gt;&lt;a href=&quot;#cb6-36&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    legend_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;Click to deselect&amp;#39;&lt;/span&gt;, &lt;/span&gt;
    &lt;span id=&quot;cb6-37&quot;&gt;&lt;a href=&quot;#cb6-37&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    legend_title_font&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;bu&quot;&gt;dict&lt;/span&gt;(size&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;16&lt;/span&gt;),&lt;/span&gt;
    &lt;span id=&quot;cb6-38&quot;&gt;&lt;a href=&quot;#cb6-38&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    legend_title_font_color&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;FireBrick&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-39&quot;&gt;&lt;a href=&quot;#cb6-39&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-40&quot;&gt;&lt;a href=&quot;#cb6-40&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-41&quot;&gt;&lt;a href=&quot;#cb6-41&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the callback to update the graph &lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-42&quot;&gt;&lt;a href=&quot;#cb6-42&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; update2(change):&lt;/span&gt;
    &lt;span id=&quot;cb6-43&quot;&gt;&lt;a href=&quot;#cb6-43&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;with&lt;/span&gt; fig2.batch_update():&lt;/span&gt;
    &lt;span id=&quot;cb6-44&quot;&gt;&lt;a href=&quot;#cb6-44&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        fig2.data[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].y &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; y2.sel(Amplitude&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;change[&lt;span class=&quot;st&quot;&gt;&amp;#39;new&amp;#39;&lt;/span&gt;]).values&lt;/span&gt;
    &lt;span id=&quot;cb6-45&quot;&gt;&lt;a href=&quot;#cb6-45&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-46&quot;&gt;&lt;a href=&quot;#cb6-46&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Link the slider and the callback&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-47&quot;&gt;&lt;a href=&quot;#cb6-47&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;A.slider.observe(update2, names&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;value&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb6-48&quot;&gt;&lt;a href=&quot;#cb6-48&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-49&quot;&gt;&lt;a href=&quot;#cb6-49&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-50&quot;&gt;&lt;a href=&quot;#cb6-50&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Display&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb6-51&quot;&gt;&lt;a href=&quot;#cb6-51&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;display(widgets.VBox([A.slider, fig2]))&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;pre&gt;&lt;code&gt;VBox(children=(FloatSlider(value=0.08, description=&amp;#39;Amplitude&amp;#39;, description_tooltip=&amp;#39;Amplitude of the resettin…&lt;/code&gt;&lt;/pre&gt;
    &lt;h2 id=&quot;cellular-automata-modeling&quot;&gt;Cellular Automata modeling&lt;/h2&gt;
    &lt;h3 id=&quot;parameters-of-the-simulation&quot;&gt;Parameters of the simulation&lt;/h3&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb8&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb8-1&quot;&gt;&lt;a href=&quot;#cb8-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;at&quot;&gt;@dataclass&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-2&quot;&gt;&lt;a href=&quot;#cb8-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;class&lt;/span&gt; Parameters:&lt;/span&gt;
    &lt;span id=&quot;cb8-3&quot;&gt;&lt;a href=&quot;#cb8-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;This class contains the parameters of the simulation.&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-4&quot;&gt;&lt;a href=&quot;#cb8-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-5&quot;&gt;&lt;a href=&quot;#cb8-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-6&quot;&gt;&lt;a href=&quot;#cb8-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;# Size of the grid&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-7&quot;&gt;&lt;a href=&quot;#cb8-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    M: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;            &lt;span class=&quot;co&quot;&gt;# number of cells in the x direction&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-8&quot;&gt;&lt;a href=&quot;#cb8-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    N: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;            &lt;span class=&quot;co&quot;&gt;# number of cells in the y direction&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-9&quot;&gt;&lt;a href=&quot;#cb8-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    fireflies_density: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;  &lt;span class=&quot;co&quot;&gt;# how much of the grid is populated&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-10&quot;&gt;&lt;a href=&quot;#cb8-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-11&quot;&gt;&lt;a href=&quot;#cb8-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;# Coupling parameters&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-12&quot;&gt;&lt;a href=&quot;#cb8-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    coupling_value: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;fl&quot;&gt;0.1&lt;/span&gt;    &lt;span class=&quot;co&quot;&gt;# float (between 0 and +/- 0.3)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-13&quot;&gt;&lt;a href=&quot;#cb8-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    neighbor_distance: &lt;span class=&quot;bu&quot;&gt;float&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;3&lt;/span&gt;  &lt;span class=&quot;co&quot;&gt;# Size of neighbourhood radius&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-14&quot;&gt;&lt;a href=&quot;#cb8-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-15&quot;&gt;&lt;a href=&quot;#cb8-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;# Simulation settings&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-16&quot;&gt;&lt;a href=&quot;#cb8-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    nb_periods: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;20&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb8-17&quot;&gt;&lt;a href=&quot;#cb8-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    time_step_per_period: &lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;100&lt;/span&gt;  &lt;span class=&quot;co&quot;&gt;# dt = 1/time_step_per_period&lt;/span&gt;&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;h3 id=&quot;simulation-script&quot;&gt;Simulation script&lt;/h3&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb9&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb9-1&quot;&gt;&lt;a href=&quot;#cb9-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; fireflies_simulator(par):&lt;/span&gt;
    &lt;span id=&quot;cb9-2&quot;&gt;&lt;a href=&quot;#cb9-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;This function simulates the fireflies system, computes the order&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-3&quot;&gt;&lt;a href=&quot;#cb9-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    parameter and shows figures of the system state during the simulation.&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-4&quot;&gt;&lt;a href=&quot;#cb9-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-5&quot;&gt;&lt;a href=&quot;#cb9-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Args:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-6&quot;&gt;&lt;a href=&quot;#cb9-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        - Parameters&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-7&quot;&gt;&lt;a href=&quot;#cb9-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-8&quot;&gt;&lt;a href=&quot;#cb9-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Returns:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-9&quot;&gt;&lt;a href=&quot;#cb9-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        The value of the order parameter over time&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-10&quot;&gt;&lt;a href=&quot;#cb9-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-11&quot;&gt;&lt;a href=&quot;#cb9-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    time_steps &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.arange(start&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb9-12&quot;&gt;&lt;a href=&quot;#cb9-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                            stop&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;par.nb_periods &lt;span class=&quot;op&quot;&gt;*&lt;/span&gt; par.time_step_per_period,&lt;/span&gt;
    &lt;span id=&quot;cb9-13&quot;&gt;&lt;a href=&quot;#cb9-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                            step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb9-14&quot;&gt;&lt;a href=&quot;#cb9-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    phases &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.zeros((time_steps.size&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;, par.N, par.M))  &lt;span class=&quot;co&quot;&gt;# 1 grid (MxN) per time step&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-15&quot;&gt;&lt;a href=&quot;#cb9-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    phases[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;] &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.random.random((par.N, par.M))          &lt;span class=&quot;co&quot;&gt;# random initial state&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-16&quot;&gt;&lt;a href=&quot;#cb9-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-17&quot;&gt;&lt;a href=&quot;#cb9-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;# empty some cells to ensure the right fireflies density&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-18&quot;&gt;&lt;a href=&quot;#cb9-18&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    nb_empty_cell &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.around((&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;par.fireflies_density)&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;par.N&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;par.M, &lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;).astype(&lt;span class=&quot;bu&quot;&gt;int&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb9-19&quot;&gt;&lt;a href=&quot;#cb9-19&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    ind &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.random.choice(phases[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].size, size&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;nb_empty_cell, replace&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;False&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb9-20&quot;&gt;&lt;a href=&quot;#cb9-20&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    phases[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].ravel()[ind] &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.nan&lt;/span&gt;
    &lt;span id=&quot;cb9-21&quot;&gt;&lt;a href=&quot;#cb9-21&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-22&quot;&gt;&lt;a href=&quot;#cb9-22&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    neighbors &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; compute_neighborhood(phases[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;], par.N, par.M, par.neighbor_distance)&lt;/span&gt;
    &lt;span id=&quot;cb9-23&quot;&gt;&lt;a href=&quot;#cb9-23&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    phase_increment &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;/&lt;/span&gt; par.time_step_per_period&lt;/span&gt;
    &lt;span id=&quot;cb9-24&quot;&gt;&lt;a href=&quot;#cb9-24&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-25&quot;&gt;&lt;a href=&quot;#cb9-25&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;for&lt;/span&gt; t &lt;span class=&quot;kw&quot;&gt;in&lt;/span&gt; time_steps:&lt;/span&gt;
    &lt;span id=&quot;cb9-26&quot;&gt;&lt;a href=&quot;#cb9-26&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        phases[t&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;] &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; phases[t] &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; phase_increment&lt;/span&gt;
    &lt;span id=&quot;cb9-27&quot;&gt;&lt;a href=&quot;#cb9-27&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        glow_idx &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.array(np.nonzero(phases[t&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;]&lt;span class=&quot;op&quot;&gt;&amp;gt;&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)).T&lt;/span&gt;
    &lt;span id=&quot;cb9-28&quot;&gt;&lt;a href=&quot;#cb9-28&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        ids &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; [np.ravel_multi_index(tup, (par.N, par.M)) &lt;span class=&quot;cf&quot;&gt;for&lt;/span&gt; tup &lt;span class=&quot;kw&quot;&gt;in&lt;/span&gt; glow_idx]&lt;/span&gt;
    &lt;span id=&quot;cb9-29&quot;&gt;&lt;a href=&quot;#cb9-29&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;for&lt;/span&gt; ind &lt;span class=&quot;kw&quot;&gt;in&lt;/span&gt; ids:&lt;/span&gt;
    &lt;span id=&quot;cb9-30&quot;&gt;&lt;a href=&quot;#cb9-30&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            i,j &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.unravel_index(ind, (par.N, par.M))&lt;/span&gt;
    &lt;span id=&quot;cb9-31&quot;&gt;&lt;a href=&quot;#cb9-31&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            phases[t&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;][neighbors[ind]] &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; nudge(phases[t&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;][neighbors[ind]], &lt;/span&gt;
    &lt;span id=&quot;cb9-32&quot;&gt;&lt;a href=&quot;#cb9-32&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                                phases[t&lt;span class=&quot;op&quot;&gt;+&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;][i,j],&lt;/span&gt;
    &lt;span id=&quot;cb9-33&quot;&gt;&lt;a href=&quot;#cb9-33&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                                                par.coupling_value)&lt;/span&gt;
    &lt;span id=&quot;cb9-34&quot;&gt;&lt;a href=&quot;#cb9-34&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-35&quot;&gt;&lt;a href=&quot;#cb9-35&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;return&lt;/span&gt;  phases&lt;/span&gt;
    &lt;span id=&quot;cb9-36&quot;&gt;&lt;a href=&quot;#cb9-36&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-37&quot;&gt;&lt;a href=&quot;#cb9-37&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-38&quot;&gt;&lt;a href=&quot;#cb9-38&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; compute_neighborhood(grid, N, M, r):&lt;/span&gt;
    &lt;span id=&quot;cb9-39&quot;&gt;&lt;a href=&quot;#cb9-39&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;For every fireflies, compute a mask array of neighboring fireflies.&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-40&quot;&gt;&lt;a href=&quot;#cb9-40&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-41&quot;&gt;&lt;a href=&quot;#cb9-41&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Args:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-42&quot;&gt;&lt;a href=&quot;#cb9-42&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        grid (ndarray): phase state of each firefly&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-43&quot;&gt;&lt;a href=&quot;#cb9-43&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        N (int): number of cells in the x direction&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-44&quot;&gt;&lt;a href=&quot;#cb9-44&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        M (int): number of cells in the y direction&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-45&quot;&gt;&lt;a href=&quot;#cb9-45&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        r (float): radius of neighbour_distance&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-46&quot;&gt;&lt;a href=&quot;#cb9-46&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-47&quot;&gt;&lt;a href=&quot;#cb9-47&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Returns:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-48&quot;&gt;&lt;a href=&quot;#cb9-48&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        dict: keys: index (ravel) of each firefly&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-49&quot;&gt;&lt;a href=&quot;#cb9-49&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;                values: mask array with neighbour fireflies&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-50&quot;&gt;&lt;a href=&quot;#cb9-50&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-51&quot;&gt;&lt;a href=&quot;#cb9-51&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    neighbors &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;bu&quot;&gt;dict&lt;/span&gt;.fromkeys((&lt;span class=&quot;bu&quot;&gt;range&lt;/span&gt;(N&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;M)), [])&lt;/span&gt;
    &lt;span id=&quot;cb9-52&quot;&gt;&lt;a href=&quot;#cb9-52&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    occupied_cells &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;~&lt;/span&gt;np.isnan(grid)&lt;/span&gt;
    &lt;span id=&quot;cb9-53&quot;&gt;&lt;a href=&quot;#cb9-53&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;for&lt;/span&gt; i &lt;span class=&quot;kw&quot;&gt;in&lt;/span&gt; &lt;span class=&quot;bu&quot;&gt;range&lt;/span&gt;(N):&lt;/span&gt;
    &lt;span id=&quot;cb9-54&quot;&gt;&lt;a href=&quot;#cb9-54&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;        &lt;span class=&quot;cf&quot;&gt;for&lt;/span&gt; j &lt;span class=&quot;kw&quot;&gt;in&lt;/span&gt; &lt;span class=&quot;bu&quot;&gt;range&lt;/span&gt;(M):&lt;/span&gt;
    &lt;span id=&quot;cb9-55&quot;&gt;&lt;a href=&quot;#cb9-55&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;cf&quot;&gt;if&lt;/span&gt; np.isnan(grid[i,j]):&lt;/span&gt;
    &lt;span id=&quot;cb9-56&quot;&gt;&lt;a href=&quot;#cb9-56&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                &lt;span class=&quot;cf&quot;&gt;continue&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-57&quot;&gt;&lt;a href=&quot;#cb9-57&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            y, x &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.ogrid[&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;i:N&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;i, &lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;j:M&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;j]&lt;/span&gt;
    &lt;span id=&quot;cb9-58&quot;&gt;&lt;a href=&quot;#cb9-58&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            mask &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; (x&lt;span class=&quot;op&quot;&gt;**&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; y&lt;span class=&quot;op&quot;&gt;**&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt; &lt;span class=&quot;op&quot;&gt;&amp;lt;=&lt;/span&gt; r&lt;span class=&quot;op&quot;&gt;**&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;)&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;occupied_cells  &lt;span class=&quot;co&quot;&gt;# only keep occupied neighboring cells&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-59&quot;&gt;&lt;a href=&quot;#cb9-59&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            &lt;span class=&quot;co&quot;&gt;# mask[i,j] = False                            # don&amp;#39;t include the cell istelf&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-60&quot;&gt;&lt;a href=&quot;#cb9-60&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;            neighbors[np.ravel_multi_index((i,j), (N,M))] &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; mask&lt;/span&gt;
    &lt;span id=&quot;cb9-61&quot;&gt;&lt;a href=&quot;#cb9-61&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;return&lt;/span&gt; neighbors&lt;/span&gt;
    &lt;span id=&quot;cb9-62&quot;&gt;&lt;a href=&quot;#cb9-62&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-63&quot;&gt;&lt;a href=&quot;#cb9-63&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-64&quot;&gt;&lt;a href=&quot;#cb9-64&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; nudge(neighbor_phases, flash_phase, amplitude):&lt;/span&gt;
    &lt;span id=&quot;cb9-65&quot;&gt;&lt;a href=&quot;#cb9-65&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;co&quot;&gt;&amp;quot;&amp;quot;&amp;quot;Nudge the neighboring fireflies.&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-66&quot;&gt;&lt;a href=&quot;#cb9-66&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-67&quot;&gt;&lt;a href=&quot;#cb9-67&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    Args:&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-68&quot;&gt;&lt;a href=&quot;#cb9-68&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        neighbor_phases (ndarray): phases of all the fireflies to nudge&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-69&quot;&gt;&lt;a href=&quot;#cb9-69&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        flash_phase (float): phase of the flashing firefly&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-70&quot;&gt;&lt;a href=&quot;#cb9-70&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;        amplitude (float): resetting strength / coupling value&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-71&quot;&gt;&lt;a href=&quot;#cb9-71&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;    &amp;quot;&amp;quot;&amp;quot;&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb9-72&quot;&gt;&lt;a href=&quot;#cb9-72&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    phase_diff &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; flash_phase &lt;span class=&quot;op&quot;&gt;-&lt;/span&gt; neighbor_phases&lt;/span&gt;
    &lt;span id=&quot;cb9-73&quot;&gt;&lt;a href=&quot;#cb9-73&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    res &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; neighbor_phases &lt;span class=&quot;op&quot;&gt;+&lt;/span&gt; amplitude&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;np.sin(&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;np.pi&lt;span class=&quot;op&quot;&gt;*&lt;/span&gt;(phase_diff))&lt;/span&gt;
    &lt;span id=&quot;cb9-74&quot;&gt;&lt;a href=&quot;#cb9-74&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    &lt;span class=&quot;cf&quot;&gt;return&lt;/span&gt; np.mod(res, &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;h3 id=&quot;simulation&quot;&gt;Simulation&lt;/h3&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb10&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb10-1&quot;&gt;&lt;a href=&quot;#cb10-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Let&amp;#39;s initiate the (default) parameters for a simulation&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb10-2&quot;&gt;&lt;a href=&quot;#cb10-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;settings &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; Parameters(N&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;20&lt;/span&gt;, M&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;20&lt;/span&gt;, &lt;/span&gt;
    &lt;span id=&quot;cb10-3&quot;&gt;&lt;a href=&quot;#cb10-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    fireflies_density&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.8&lt;/span&gt;, &lt;/span&gt;
    &lt;span id=&quot;cb10-4&quot;&gt;&lt;a href=&quot;#cb10-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    coupling_value&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.01&lt;/span&gt;, &lt;/span&gt;
    &lt;span id=&quot;cb10-5&quot;&gt;&lt;a href=&quot;#cb10-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    neighbor_distance&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb10-6&quot;&gt;&lt;a href=&quot;#cb10-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    time_step_per_period&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;100&lt;/span&gt;, &lt;/span&gt;
    &lt;span id=&quot;cb10-7&quot;&gt;&lt;a href=&quot;#cb10-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    nb_periods&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;100&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb10-8&quot;&gt;&lt;a href=&quot;#cb10-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb10-9&quot;&gt;&lt;a href=&quot;#cb10-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb10-10&quot;&gt;&lt;a href=&quot;#cb10-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# And let&amp;#39;s run the simulation&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb10-11&quot;&gt;&lt;a href=&quot;#cb10-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;heatmaps &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; fireflies_simulator(settings)&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb11&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb11-1&quot;&gt;&lt;a href=&quot;#cb11-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Let&amp;#39;s visualiza the simulation&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-2&quot;&gt;&lt;a href=&quot;#cb11-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-3&quot;&gt;&lt;a href=&quot;#cb11-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;mymin &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-4&quot;&gt;&lt;a href=&quot;#cb11-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;mymax &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; heatmaps.shape[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;]&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-5&quot;&gt;&lt;a href=&quot;#cb11-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;mystep &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; &lt;span class=&quot;dv&quot;&gt;10&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-6&quot;&gt;&lt;a href=&quot;#cb11-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-7&quot;&gt;&lt;a href=&quot;#cb11-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-8&quot;&gt;&lt;a href=&quot;#cb11-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the figure&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-9&quot;&gt;&lt;a href=&quot;#cb11-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig5 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.FigureWidget(&lt;/span&gt;
    &lt;span id=&quot;cb11-10&quot;&gt;&lt;a href=&quot;#cb11-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    data&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;go.Heatmap(z&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;heatmaps[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;], colorscale&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;Hot&amp;#39;&lt;/span&gt;, reversescale&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;va&quot;&gt;True&lt;/span&gt;),&lt;/span&gt;
    &lt;span id=&quot;cb11-11&quot;&gt;&lt;a href=&quot;#cb11-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    layout&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;go.Layout(title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;Fireflies Simulator&amp;quot;&lt;/span&gt;))&lt;/span&gt;
    &lt;span id=&quot;cb11-12&quot;&gt;&lt;a href=&quot;#cb11-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig5.data[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].update(zmin&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;, zmax&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;)  &lt;span class=&quot;co&quot;&gt;# fix the colorscale to [0,1]&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-13&quot;&gt;&lt;a href=&quot;#cb11-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-14&quot;&gt;&lt;a href=&quot;#cb11-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-15&quot;&gt;&lt;a href=&quot;#cb11-15&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Set the callback to update the graph &lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-16&quot;&gt;&lt;a href=&quot;#cb11-16&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;kw&quot;&gt;def&lt;/span&gt; update5(change):&lt;/span&gt;
    &lt;span id=&quot;cb11-17&quot;&gt;&lt;a href=&quot;#cb11-17&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;    fig5.data[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;].z &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; heatmaps[change[&lt;span class=&quot;st&quot;&gt;&amp;#39;new&amp;#39;&lt;/span&gt;]]&lt;/span&gt;
    &lt;span id=&quot;cb11-18&quot;&gt;&lt;a href=&quot;#cb11-18&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-19&quot;&gt;&lt;a href=&quot;#cb11-19&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-20&quot;&gt;&lt;a href=&quot;#cb11-20&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Create the animation display and link it to the callback function&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-21&quot;&gt;&lt;a href=&quot;#cb11-21&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;myanimation &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; AnimateSlider(start&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;mymin, stop&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;mymax, step&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;mystep)&lt;/span&gt;
    &lt;span id=&quot;cb11-22&quot;&gt;&lt;a href=&quot;#cb11-22&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;myanimation.slider.observe(update5, names&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;value&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb11-23&quot;&gt;&lt;a href=&quot;#cb11-23&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;controllers &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; widgets.HBox([myanimation.play, myanimation.slider])&lt;/span&gt;
    &lt;span id=&quot;cb11-24&quot;&gt;&lt;a href=&quot;#cb11-24&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-25&quot;&gt;&lt;a href=&quot;#cb11-25&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-26&quot;&gt;&lt;a href=&quot;#cb11-26&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Display&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb11-27&quot;&gt;&lt;a href=&quot;#cb11-27&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;display(widgets.VBox([controllers, fig5]))&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;pre&gt;&lt;code&gt;VBox(children=(HBox(children=(Play(value=0, description=&amp;#39;Press play&amp;#39;, max=10000, step=10), IntSlider(value=0, …&lt;/code&gt;&lt;/pre&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb13&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb13-1&quot;&gt;&lt;a href=&quot;#cb13-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;# Statistical results&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb13-2&quot;&gt;&lt;a href=&quot;#cb13-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb13-3&quot;&gt;&lt;a href=&quot;#cb13-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;t3 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.arange(heatmaps.shape[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;])&lt;/span&gt;
    &lt;span id=&quot;cb13-4&quot;&gt;&lt;a href=&quot;#cb13-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;avg &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.nanmean(heatmaps, axis&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;(&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;,&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;))&lt;/span&gt;
    &lt;span id=&quot;cb13-5&quot;&gt;&lt;a href=&quot;#cb13-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;std &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; np.nanstd(heatmaps, axis&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;(&lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;,&lt;span class=&quot;dv&quot;&gt;2&lt;/span&gt;))&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;div class=&quot;sourceCode&quot; id=&quot;cb14&quot;&gt;&lt;pre class=&quot;sourceCode python&quot;&gt;&lt;code class=&quot;sourceCode python&quot;&gt;&lt;span id=&quot;cb14-1&quot;&gt;&lt;a href=&quot;#cb14-1&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;span class=&quot;co&quot;&gt;#show light intensity over time (the last 500 time steps)&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb14-2&quot;&gt;&lt;a href=&quot;#cb14-2&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb14-3&quot;&gt;&lt;a href=&quot;#cb14-3&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;trace0 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.Scatter(x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;t3[&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;:], y&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;avg[&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;:], name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;average&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb14-4&quot;&gt;&lt;a href=&quot;#cb14-4&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;trace1 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.Scatter(x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;t3[&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;:], y&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;std[&lt;span class=&quot;op&quot;&gt;-&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;1000&lt;/span&gt;:], name&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;std dev&amp;#39;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb14-5&quot;&gt;&lt;a href=&quot;#cb14-5&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;&lt;/span&gt;
    &lt;span id=&quot;cb14-6&quot;&gt;&lt;a href=&quot;#cb14-6&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig3 &lt;span class=&quot;op&quot;&gt;=&lt;/span&gt; go.FigureWidget([trace0, trace1])&lt;/span&gt;
    &lt;span id=&quot;cb14-7&quot;&gt;&lt;a href=&quot;#cb14-7&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig3.update_yaxes(&lt;span class=&quot;bu&quot;&gt;range&lt;/span&gt;&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;[&lt;span class=&quot;dv&quot;&gt;0&lt;/span&gt;, &lt;span class=&quot;dv&quot;&gt;1&lt;/span&gt;])       &lt;span class=&quot;co&quot;&gt;# fix the scale to [0,1] for easy comparison&lt;/span&gt;&lt;/span&gt;
    &lt;span id=&quot;cb14-8&quot;&gt;&lt;a href=&quot;#cb14-8&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig3.update_layout(template&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;#39;none&amp;#39;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb14-9&quot;&gt;&lt;a href=&quot;#cb14-9&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    width&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;800&lt;/span&gt;, height&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;dv&quot;&gt;500&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb14-10&quot;&gt;&lt;a href=&quot;#cb14-10&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;Light intensity&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb14-11&quot;&gt;&lt;a href=&quot;#cb14-11&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    title_x&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;fl&quot;&gt;0.5&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb14-12&quot;&gt;&lt;a href=&quot;#cb14-12&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    xaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;t&amp;quot;&lt;/span&gt;,&lt;/span&gt;
    &lt;span id=&quot;cb14-13&quot;&gt;&lt;a href=&quot;#cb14-13&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;                    yaxis_title&lt;span class=&quot;op&quot;&gt;=&lt;/span&gt;&lt;span class=&quot;st&quot;&gt;&amp;quot;I&amp;quot;&lt;/span&gt;)&lt;/span&gt;
    &lt;span id=&quot;cb14-14&quot;&gt;&lt;a href=&quot;#cb14-14&quot; aria-hidden=&quot;true&quot; tabindex=&quot;-1&quot;&gt;&lt;/a&gt;fig3&lt;/span&gt;&lt;/code&gt;&lt;/pre&gt;&lt;/div&gt;
    &lt;pre&gt;&lt;code&gt;FigureWidget({
        &amp;#39;data&amp;#39;: [{&amp;#39;name&amp;#39;: &amp;#39;average&amp;#39;,
                  &amp;#39;type&amp;#39;: &amp;#39;scatter&amp;#39;,
                  &amp;#39;uid&amp;#39;: &amp;#39;f855cbb…&lt;/code&gt;&lt;/pre&gt;
    &lt;!--bibtex
    
    @Article{PER-GRA:2007,
      Author    = {P\&#39;erez, Fernando and Granger, Brian E.},
      Title     = {{IP}ython: a System for Interactive Scientific Computing},
      Journal   = {Computing in Science and Engineering},
      Volume    = {9},
      Number    = {3},
      Pages     = {21--29},
      month     = may,
      year      = 2007,
      url       = &quot;http://ipython.org&quot;,
      ISSN      = &quot;1521-9615&quot;,
      doi       = {10.1109/MCSE.2007.53},
      publisher = {IEEE Computer Society},
    }
    
    @article{Papa2007,
      author = {Papa, David A. and Markov, Igor L.},
      journal = {Approximation algorithms and metaheuristics},
      pages = {1--38},
      title = {{Hypergraph partitioning and clustering}},
      url = {http://www.podload.org/pubs/book/part\_survey.pdf},
      year = {2007}
    }
    
    --&gt;
    &lt;p&gt;Examples of citations: &lt;a href=&quot;#cite-PER-GRA:2007&quot;&gt;CITE&lt;/a&gt; or &lt;a href=&quot;#cite-Papa2007&quot;&gt;CITE&lt;/a&gt;.&lt;/p&gt;
    &lt;!--
    
    
    ```python
    #automatic document conversion to markdown and then to word
    #first convert the ipython notebook paper.ipynb to markdown
    os.system(&quot;jupyter nbconvert --to markdown Fireflies-Synchronization.ipynb&quot;)
        
     #next convert markdown to ms word
    conversion = f&quot;pandoc -s Fireflies-Synchronization.md --filter --citeproc CITATIONS --bibliography Fireflies-Synchronization.bib --csl=apa.csl&quot;
    
    os.system(conversion)
    ```
    
        [NbConvertApp] Converting notebook Fireflies-Synchronization.ipynb to markdown
        [NbConvertApp] Writing 17431 bytes to Fireflies-Synchronization.md
        [WARNING] Could not deduce format from file extension 
          Defaulting to markdown
        pandoc: Fireflies-Synchronization: openBinaryFile: does not exist (No such file or directory)
    
    
    
    
    
        256
    
    
    
    --&gt;
    &lt;/body&gt;
    &lt;/html&gt;
    
    
    [WARNING] Could not convert TeX math \begin{align*}
      &amp;\theta_{t+1} =  \theta_t + \frac{1}{T} &amp;&amp; \text{ (mod 1)} \\
      \Rightarrow&amp; \theta_{t+1} - \theta_t = \frac{1}{T} &amp;&amp;  \text{ (mod 1)} \\
      \Rightarrow&amp; \frac{\theta_{t+1} - \theta_t}{(t+1) - t} = \frac{1}{T} &amp;&amp; \text{ (mod 1)} \\
      \Rightarrow&amp; \theta(t) = \frac{t}{T} &amp;&amp; \text{ (mod 1)}
      \end{align*}, rendering as TeX
    [WARNING] This document format requires a nonempty &lt;title&gt; element.
      Defaulting to &#39;Fireflies-Synchronization&#39; as the title.
      To specify a title, use &#39;title&#39; in metadata or --metadata title=&quot;...&quot;.
    
    
    
    
    
    0</code></pre>
    <p>–&gt;</p>
    </body>
    </html>


-->
