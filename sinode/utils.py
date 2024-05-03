import jax
import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context="notebook", style="ticks",
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
plt.style.use("dark_background")


def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds



## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, context="notebook", style="ticks", dark_background=False, **kwargs):
    sns.set_theme(context=context, style=style,
            font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
    if dark_background:
        plt.style.use("dark_background")

    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax



def RK4(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """ Perform numerical integration with a time step divided by the evaluation subdivision factor (Not necessarily equally spaced). If we get NaNs, we can try to increasing the subdivision factor for finer time steps."""
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        k1 = h * fun(t_prev, y_prev, *args)
        k2 = h * fun(t_prev + h/2., y_prev + k1/2., *args)
        k3 = h * fun(t_prev + h/2., y_prev + k2/2., *args)
        k4 = h * fun(t + h, y_prev + k3, *args)
        y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]
