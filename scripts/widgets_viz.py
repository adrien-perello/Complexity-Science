"""Some math widget visualisation scripts"""

import threading
from numbers import Integral
from fractions import Fraction
from dataclasses import dataclass
from ast import literal_eval

# from math import prod
import numpy as np
import xarray as xr
import ipywidgets as ipw


############################
#     GENERAL FUNCTIONS
############################


def get_frac(step, readout_format=".16f", atol=1e-12):
    """Convert decimal number into fraction of integers
    + raise warning if potential irrational number
    """
    precision = "{:" + readout_format + "}"
    frac = Fraction(precision.format(step))
    if frac.denominator > 1 / atol:
        print(
            "WARNING: potential Floats inconsistencies."
            " Check if 'step' is an irrational number"
        )
    return (frac.numerator, frac.denominator)


def nparange(start, stop, step):
    """Modified np.arange()
        - improve float precision (by use of fractions)
        - includes endpoint

    Args:
        start, stop, step: float (stop is included in array)

    Returns:
        ndarray
    """
    delta, zoom = get_frac(step)
    return np.arange(start * zoom, stop * zoom + delta, delta) / zoom


def lst_depth(lst):
    """Check max depth of nested list."""
    if isinstance(lst, list):
        return 1 + max(lst_depth(item) for item in lst)
    return 0


def debounce(wait):
    """(https://gist.github.com/walkermatt/2871026)
    Decorator that will postpone a function's
    execution until after `wait` seconds
    have elapsed since the last time it was invoked.
    """

    def decorator(function):
        def debounced(*args, **kwargs):
            def call_function():
                debounced.timer = None
                return function(*args, **kwargs)

            if debounced.timer is not None:
                debounced.timer.cancel()
            debounced.timer = threading.Timer(wait, call_function)
            debounced.timer.start()

        debounced.timer = None
        return debounced

    return decorator


def get_mat_str(arr):
    """Get Latex (Matrix) representation of an array.

    Args:
        arr (number): Matrix as a nested list

    Returns:
        str: raw string represention of array
    """
    temp = (" & ".join(str(col) for col in row) for row in arr)
    arrstr = " \\ ".join(row for row in temp)
    return (
        r"$\begin{bmatrix} "
        + arrstr.encode("unicode_escape").decode()
        + r" \end{bmatrix}$"
    )


def matrix_repr(arr, name="", text=""):
    """Get representation of a matrix (nested list) as ipw.HTMLMath.

    Args:
        arr (number): Matrix as nested list
        name (str, optional): Name of the widget
        text (str, optional): Label of the matrix

    Returns:
        ipw.HTMLMath: ipywidgets
    """
    return ipw.HTMLMath(
        value=get_mat_str(arr),
        placeholder=name,
        description=text,
    )


############################
#         CLASSES
############################
@dataclass
class Slider:
    """Represent a range of (linear) values as both:
    - an np.array
    - an ipywidget.Floatslider
    """

    name: str
    settings: tuple  #  (start, end, stop [, val_ini])
    hist_delay: float = 1

    def __post_init__(self):
        start, stop, step, *val_ini = self.settings
        self.arr = nparange(start, stop, step)
        self.val = val_ini[0] if val_ini else np.random.choice(self.arr, 1)[0]
        self.hist = [self.val]
        self.slider = ipw.FloatSlider(
            min=start,
            max=stop,
            step=step,
            value=self.val,
            description=self.name,
            continuous_update=True,
        )
        self._link_val_slider()

    def __getitem__(self, key):
        """Get val[i]"""
        return self.arr[key]

    def _update_val(self, change):
        """Update val"""
        self.val = change["new"]

    def _update_hist(self, change):
        """Update hist.
        Method decorated with @debounce(hist_delay)"""
        self.hist.append(change["new"])

    def _link_val_slider(self):
        """Link sliders value with matrix repr"""
        self._update_hist = debounce(self.hist_delay)(
            self._update_hist
        )  # = @debounce(hist_delay)
        self.slider.observe(self._update_val, names="value")
        self.slider.observe(self._update_hist, names="value")


# TO UPDATE (to tuple)
# @dataclass
# class LogSlider:
#     """Represent a range of log values as both:
#     - an np.array
#     - an ipywidget.FloatLogSlider
#     """

#     name: str
#     start: float
#     stop: float
#     num: int
#     val_ini: float = None
#     base: int = 10
#     decimals: int = 1

#     def __post_init__(self):
#         # create numpy array of all values
#         self.val = np.around(
#             np.logspace(start=self.start, stop=self.stop, num=self.num, endpoint=True),
#             self.decimals,
#         )
#         # check each value is unique
#         if self.val.size != np.unique(self.val, return_counts=False).size:
#             print(
#                 f"WARNING: Repeated values in {self.name}.val"
#                 ", increase 'decimals' or reduce 'num'"
#             )

#         # pick initial value if not provided
#         if not self.val_ini:
#             self.val_ini = np.random.choice(self.val, 1)[0]

#         # convert num into step for FloatLogSlider
#         step = (self.stop - self.start) / (self.num - 1)

#         # create slider
#         self.slider = ipw.FloatLogSlider(
#             min=self.start,
#             max=self.stop,
#             step=step,
#             value=self.val_ini,
#             base=self.base,
#             description=self.name,
#             readout_format=f".{self.decimals}f",
#         )

# TO UPDATE
# @dataclass
# class AnimateSlider:
#     """Represents an animation controllers with:
#     - an ipywidget.Play (play, pause, stop)
#     - an integer slider
#     """

#     start: int
#     stop: int
#     step: float
#     name: str = "Press play"
#     val_ini: float = None
#     interval: float = 100  # time interval (in ms)

#     def __post_init__(self):
#         # create the play widget
#         self.play = ipw.Play(
#             min=self.start,
#             max=self.stop,
#             step=self.step,
#             interval=self.interval,
#             description=self.name,
#         )

#         if self.val_ini:
#             self.play.value = self.val_ini

#         # create a slider for visualization
#         self.slider = ipw.IntSlider(min=self.start, max=self.stop, step=self.step)

#         # Link the slider and the play widget
#         ipw.jslink((self.play, "value"), (self.slider, "value"))


# TO INHERIT FROM MATRIXSLIDER
@dataclass
class VectorSlider2D:
    """Represent a 2D vector as 2 linear sliders (one for each dimension).

    Args:
        - str for name
        - 2x tuples (start, end, step) for both x and y

    Returns:
        - 2x ndarray of all options of both x and y
        - 2x dict() for value, index mapping
        - 2x sliders
    """

    name: str
    x_settings: tuple  # (start, end, stop, val_ini)
    y_settings: tuple  # (start, end, stop, val_ini)

    def __post_init__(self):
        self.x = Slider(self.name + "x", *self.x_settings)
        self.y = Slider(self.name + "y", *self.y_settings)
        self.slider = ipw.VBox([self.y.slider, self.x.slider])
        self.y.slider.orientation = "vertical"
        self.slider.layout.align_items = "center"

    def __getitem__(self, coord):
        """Get Vector[i,j]"""
        i, j = coord
        return np.array([self.x[i], self.y[j]])

    def __add__(self, other):
        """Get all vector combinations from adding 2 vectors"""
        datax = xr.DataArray(
            self.x[:, None] + other.x[:],
            dims=(self.name, other.name),
            coords={self.name: self.x[:], other.name: other.x[:]},
        )
        datay = xr.DataArray(
            self.y[:, None] + other.y[:],
            dims=(self.name, other.name),
            coords={self.name: self.y[:], other.name: other.y[:]},
        )
        return xr.Dataset(
            data_vars={"x": datax, "y": datay},
            attrs={"name": f"{self.name}+{other.name}"},
        )

    def __sub__(self, other):
        """Get all vector combinations from subtracting 2 vectors"""
        datax = xr.DataArray(
            self.x[:, None] - other.x[:],
            dims=(self.name, other.name),
            coords={self.name: self.x[:], other.name: other.x[:]},
        )
        datay = xr.DataArray(
            self.y[:, None] - other.y[:],
            dims=(self.name, other.name),
            coords={self.name: self.y[:], other.name: other.y[:]},
        )
        return xr.Dataset(
            data_vars={"x": datax, "y": datay},
            attrs={"name": f"{self.name}-{other.name}"},
        )

    def __mul__(self, scalar):
        if not np.isscalar(scalar):
            raise Exception(f"{scalar} is not a scalar")

        name = f"{self.name} *{str(scalar)}"
        x_settings = self.x_settings + (self.x.val,)
        y_settings = self.y_settings + (self.y.val,)
        return VectorSlider2D(
            name, scalar * np.array(x_settings), scalar * np.array(y_settings)
        )

    def __rmul__(self, scalar):
        return self * scalar

    def get_val(self):
        """Get val[i]"""
        return np.array([self.x.val, self.y.val])


@dataclass
class SliderMatrix:
    """..."""

    name: str
    settings: list  # list of (start, end, stop [, val_ini])
    hist_delay: float = 1

    def __post_init__(self):
        self._check_dim()
        self._init_size_and_shape()
        self._init_repr()
        self._init_sliders_and_val()
        self._link_val_slider()

    def _check_dim(self):
        depth = lst_depth(self.settings)
        if depth == 0:  # row vector
            raise TypeError(
                "This is a simple slider, not a MatrixSlider.\
                        Use appropriate class"
            )
        if depth == 1:
            self.settings = [self.settings]

    def _init_size_and_shape(self):
        self.shape = (len(self.settings), len(self.settings[0]))
        # self.size = prod(self.shape)

    def _init_repr(self):
        self.repr = ipw.GridspecLayout(
            self.shape[0],
            1 + self.shape[1],
            layout=ipw.Layout(
                # grid_template_columns="20% 40% 40%",
                # align_items="center",
                # justify_items="flex-start",
                # width="80%"
                # align_self="center",
            ),
        )

    def _init_sliders_and_val(self):
        self.val = np.zeros(shape=self.shape)
        self.sliders = [[] for _ in range(self.shape[0])]
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.sliders[i].append(
                    Slider(f"{self.name}({str(i)},{str(j)})", self.settings[i][j])
                )
                self.val[i, j] = self.sliders[i][j].val
                self.repr[i, 1 + j] = self.sliders[i][j].slider
        self.repr[:, 0] = matrix_repr(self.val, name=self.name, text=f"{self.name} =")
        self.hist = [self.val]

    def __len__(self):
        return self.shape[0]

    def _getrow(self, idx):
        return self.sliders[idx]

    def __getitem__(self, idx):
        if isinstance(idx, (Integral, slice)):
            return self._getrow(idx)
        rowidx, colidx = idx
        if isinstance(rowidx, Integral):
            return self._getrow(rowidx)[colidx]
        # else:
        return [row[colidx] for row in self._getrow(rowidx)]

    def _update_val(self, change):
        """Update val"""
        key = change["owner"].description.split(self.name)[-1]
        self.val[literal_eval(key)] = change["new"]
        self.repr[0, 0].value = get_mat_str(self.val)

    def _update_hist(self, change):
        """Update hist.
        Method decorated with @debounce(hist_delay)"""
        self.hist.append(change["new"])

    def _link_val_slider(self):
        """Link sliders value with matrix repr"""
        self._update_hist = debounce(self.hist_delay)(
            self._update_hist
        )  # = @debounce(hist_delay)
        for slider in np.array(self.sliders).ravel():
            slider.slider.observe(self._update_val, names="value")
            slider.slider.observe(self._update_hist, names="value")

    def show(self):
        """Display matrix repr"""
        return self.repr
