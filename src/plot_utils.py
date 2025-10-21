import matplotlib.pyplot as plt
import os.path as osp
import json
import os
import numpy as np
import pandas
from collections import defaultdict, namedtuple
from baselines.bench import monitor
from baselines.logger import read_json, read_csv

def smooth(y, radius, mode='two_sided', valid_only=False):
    '''
    Smooth signal y, where radius is determines the size of the window

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]

    valid_only: put nan in entries where the full-sized window is not available

    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out

def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''

    low = xolds[0] if low is None else low
    high = xolds[-1] if high is None else high

    assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
    assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
    assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))


    xolds = xolds.astype('float64')
    yolds = yolds.astype('float64')

    luoi = 0 # last unused old index
    sum_y = 0.
    count_y = 0.
    xnews = np.linspace(low, high, n)
    decay_period = (high - low) / (n - 1) * decay_steps
    interstep_decay = np.exp(- 1. / decay_steps)
    sum_ys = np.zeros_like(xnews)
    count_ys = np.zeros_like(xnews)
    for i in range(n):
        xnew = xnews[i]
        sum_y *= interstep_decay
        count_y *= interstep_decay
        while True:
            if luoi >= len(xolds):
                break
            xold = xolds[luoi]
            if xold <= xnew:
                decay = np.exp(- (xnew - xold) / decay_period)
                sum_y += decay * yolds[luoi]
                count_y += decay
                luoi += 1
            else:
                break
        sum_ys[i] = sum_y
        count_ys[i] = count_y

    ys = sum_ys / count_ys
    ys[count_ys < low_counts_threshold] = np.nan

    return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1., low_counts_threshold=1e-8):
    '''
    perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]

    Arguments:

    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds

    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]

    n: int                - number of points in new x grid

    decay_steps: float    - EMA decay factor, expressed in new x grid steps.

    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN

    Returns:
        tuple sum_ys, count_ys where
            xs        - array with new x grid
            ys        - array of EMA of y at each point of the new x grid
            count_ys  - array of EMA of y counts at each point of the new x grid

    '''
    xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold=0)
    _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold=0)
    ys2 = ys2[::-1]
    count_ys2 = count_ys2[::-1]
    count_ys = count_ys1 + count_ys2
    ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
    ys[count_ys < low_counts_threshold] = np.nan
    return xs, ys, count_ys

Result = namedtuple('Result', 'monitor progress dirname metadata')
Result.__new__.__defaults__ = (None,) * len(Result._fields)

def load_results(root_dir_or_dirs, enable_progress=True, enable_monitor=True, verbose=False):
    '''
    load summaries of runs from a list of directories (including subdirectories)
    Arguments:

    enable_progress: bool - if True, will attempt to load data from progress.csv files (data saved by logger). Default: True

    enable_monitor: bool - if True, will attempt to load data from monitor.csv files (data saved by Monitor environment wrapper). Default: True

    verbose: bool - if True, will print out list of directories from which the data is loaded. Default: False


    Returns:
    List of Result objects with the following fields:
         - dirname - path to the directory data was loaded from
         - metadata - run metadata (such as command-line arguments and anything else in metadata.json file
         - monitor - if enable_monitor is True, this field contains pandas dataframe with loaded monitor.csv file (or aggregate of all *.monitor.csv files in the directory)
         - progress - if enable_progress is True, this field contains pandas dataframe with loaded progress.csv file
    '''
    import re
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist"%rootdir
        for dirname, dirs, files in os.walk(rootdir):
            if '-proc' in dirname:
                files[:] = []
                continue
            monitor_re = re.compile(r'(\d+\.)?(\d+\.)?monitor\.csv')
            if set(['metadata.json', 'monitor.json', 'progress.json', 'progress.csv']).intersection(files) or \
               any([f for f in files if monitor_re.match(f)]):  # also match monitor files like 0.1.monitor.csv
                # used to be uncommented, which means do not go deeper than current directory if any of the data files
                # are found
                # dirs[:] = []
                result = {'dirname' : dirname}
                if "metadata.json" in files:
                    with open(osp.join(dirname, "metadata.json"), "r") as fh:
                        result['metadata'] = json.load(fh)
                progjson = osp.join(dirname, "progress.json")
                progcsv = osp.join(dirname, "progress.csv")
                if enable_progress:
                    if osp.exists(progjson):
                        result['progress'] = pandas.DataFrame(read_json(progjson))
                    elif osp.exists(progcsv):
                        try:
                            result['progress'] = read_csv(progcsv)
                        except pandas.errors.EmptyDataError:
                            print('skipping progress file in ', dirname, 'empty data')
                    else:
                        if verbose: print('skipping %s: no progress file'%dirname)

                if enable_monitor:
                    try:
                        result['monitor'] = pandas.DataFrame(monitor.load_results(dirname))
                    except monitor.LoadMonitorResultsError:
                        print('skipping %s: no monitor files'%dirname)
                    except Exception as e:
                        print('exception loading monitor file in %s: %s'%(dirname, e))

                if result.get('monitor') is not None or result.get('progress') is not None:
                    allresults.append(Result(**result))
                    if verbose:
                        print('successfully loaded %s'%dirname)

    if verbose: print('loaded %i results'%len(allresults))
    return allresults

COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkgreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']


def default_xy_fn(r):
    x = np.cumsum(r.monitor.l)
    y = smooth(r.monitor.r, radius=10)
    return x,y

def default_split_fn(r):
    import re
    # match name between slash and -<digits> at the end of the string
    # (slash in the beginning or -<digits> in the end or either may be missing)
    match = re.search(r'[^/-]+(?=(-\d+)?\Z)', r.dirname)
    if match:
        return match.group(0)

import numpy as np
import matplotlib.pyplot as plt

def plot_results(
    x,
    y,
    ax=None,
    figsize=(10, 6),
    xlabel=None,
    ylabel=None,
    title=None,
    legend_label=None,
    shaded_std=False,
    shaded_err=False,
    smooth_step=1.0,
    resample=0,
    line_style='-',
    color=None,
    grid=False,
):
    '''
    Plot y versus x, optionally on a provided Axes object.
    
    Parameters:
    -----------
    x : list or numpy array
        The x-coordinates of the data points.
    
    y : list or numpy array
        The y-coordinates of the data points.
    
    ax : matplotlib Axes object, optional
        The axes on which to plot. If None, creates a new figure and axes.
    
    figsize : tuple of two integers, optional
        Figure size in inches. Only used if ax is None. Default is (10, 6).
    
    xlabel : str, optional
        Label for the x-axis.
    
    ylabel : str, optional
        Label for the y-axis.
    
    title : str, optional
        Title of the plot.
    
    legend_label : str, optional
        Label for the line in the legend.
    
    shaded_std : bool, optional
        If True, shade the region representing one standard deviation from the mean.
        Default is False.
    
    shaded_err : bool, optional
        If True, shade the region representing the standard error.
        Default is False.
    
    smooth_step : float, optional
        The smoothing parameter for the symmetric exponential moving average.
        Only applicable if resample > 0.
        Default is 1.0.
    
    resample : int, optional
        If greater than zero, resample the data onto a new grid with `resample` points.
        Useful for smoothing the data.
        Default is 0 (no resampling).
    
    line_style : str, optional
        Line style for the plot (e.g., '-', '--', '-.', ':').
        Default is '-'.
    
    color : str, optional
        Color of the line and shaded regions.
        Default is 'b' (blue).
    
    grid : bool, optional
        If True, display a grid on the plot.
        Default is False.
    
    Returns:
    --------
    ax : matplotlib Axes
        The axes object with the plot.
    '''
    
    x = np.asarray(x)
    y = np.asarray(y)

    # Resample and smooth the data if needed
    if resample > 0:
        from scipy.interpolate import interp1d

        # Define the new x-axis for resampling
        x_new = np.linspace(x.min(), x.max(), resample)
        
        # Interpolate y-values onto the new x-axis
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        y_new = f(x_new)
        
        # Apply smoothing using symmetric exponential moving average
        def symmetric_ema(y, decay_steps):
            alpha = 2 / (decay_steps + 1)
            y_forward = np.copy(y)
            y_backward = np.copy(y)
            # Forward pass
            for i in range(1, len(y)):
                y_forward[i] = alpha * y[i] + (1 - alpha) * y_forward[i - 1]
            # Backward pass
            for i in range(len(y) - 2, -1, -1):
                y_backward[i] = alpha * y[i] + (1 - alpha) * y_backward[i + 1]
            y_smooth = (y_forward + y_backward) / 2
            return y_smooth

        y_smooth = symmetric_ema(y_new, smooth_step)
        x = x_new
        y = y_smooth

    # Create the plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None  # Figure is not created by this function
    
    ax.plot(x, y, line_style, color=color, label=legend_label)

    # Shaded regions for standard deviation or standard error
    if shaded_std or shaded_err:
        if shaded_std:
            std_dev = np.std(y)
            ax.fill_between(x, y - std_dev, y + std_dev, color=color, alpha=0.2)
        if shaded_err:
            std_err = np.std(y) / np.sqrt(len(y))
            ax.fill_between(x, y - std_err, y + std_err, color=color, alpha=0.2)

    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True)

    # Show legend if label is provided
    if legend_label:
        ax.legend()

    if fig is not None:
        plt.tight_layout()
        plt.show()

    return ax

def regression_analysis(df):
    xcols = list(df.columns.copy())
    xcols.remove('score')
    ycols = ['score']
    import statsmodels.api as sm
    mod = sm.OLS(df[ycols], sm.add_constant(df[xcols]), hasconst=False)
    res = mod.fit()
    print(res.summary())

def test_smooth():
    norig = 100
    nup = 300
    ndown = 30
    xs = np.cumsum(np.random.rand(norig) * 10 / norig)
    yclean = np.sin(xs)
    ys = yclean + .1 * np.random.randn(yclean.size)
    xup, yup, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), nup, decay_steps=nup/ndown)
    xdown, ydown, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), ndown, decay_steps=ndown/ndown)
    xsame, ysame, _ = symmetric_ema(xs, ys, xs.min(), xs.max(), norig, decay_steps=norig/ndown)
    plt.plot(xs, ys, label='orig', marker='x')
    plt.plot(xup, yup, label='up', marker='x')
    plt.plot(xdown, ydown, label='down', marker='x')
    plt.plot(xsame, ysame, label='same', marker='x')
    plt.plot(xs, yclean, label='clean', marker='x')
    plt.legend()
    plt.show()

