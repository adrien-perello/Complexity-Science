"""Some math widget visualisation scripts"""


from fractions import Fraction
from dataclasses import dataclass
import numpy as np
import xarray as xr
import ipywidgets as widgets


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

        self.slider = widgets.FloatSlider(
            min=self.start,
            max=self.stop,
            step=self.step,
            value=self.val_ini,
            description=self.name,
            continuous_update=True,
        )

    def __getitem__(self, key):
        """Get val[i]"""
        return self.val[key]

    def get_val(self):
        """Get val[i]"""
        return self.slider.value


@dataclass
class LogSlider:
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
        self.val = np.around(
            np.logspace(start=self.start, stop=self.stop, num=self.num, endpoint=True),
            self.decimals,
        )
        # check each value is unique
        if self.val.size != np.unique(self.val, return_counts=False).size:
            print(
                f"WARNING: Repeated values in {self.name}.val"
                ", increase 'decimals' or reduce 'num'"
            )

        # pick initial value if not provided
        if not self.val_ini:
            self.val_ini = np.random.choice(self.val, 1)[0]

        # convert num into step for FloatLogSlider
        step = (self.stop - self.start) / (self.num - 1)

        # create slider
        self.slider = widgets.FloatLogSlider(
            min=self.start,
            max=self.stop,
            step=step,
            value=self.val_ini,
            base=self.base,
            description=self.name,
            readout_format=f".{self.decimals}f",
        )


@dataclass
class AnimateSlider:
    """Represents an animation controllers with:
    - an ipywidget.Play (play, pause, stop)
    - an integer slider
    """

    start: int
    stop: int
    step: float
    name: str = "Press play"
    val_ini: float = None
    interval: float = 100  # time interval (in ms)

    def __post_init__(self):
        # create the play widget
        self.play = widgets.Play(
            min=self.start,
            max=self.stop,
            step=self.step,
            interval=self.interval,
            description=self.name,
        )

        if self.val_ini:
            self.play.value = self.val_ini

        # create a slider for visualization
        self.slider = widgets.IntSlider(min=self.start, max=self.stop, step=self.step)

        # Link the slider and the play widget
        widgets.jslink((self.play, "value"), (self.slider, "value"))


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
        self.slider = widgets.VBox([self.y.slider, self.x.slider])
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

    def get_val(self):
        """Get val[i]"""
        return np.array([self.x.get_val(), self.y.get_val()])


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
