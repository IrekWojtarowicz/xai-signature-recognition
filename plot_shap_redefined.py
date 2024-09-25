import json
import random
import string
from typing import Optional

import matplotlib.pyplot as pl
import numpy as np
from matplotlib.colors import Colormap

try:
    from IPython.display import HTML, display
    have_ipython = True
except ImportError:
    have_ipython = False

def plot_shap(shap_values: Explanation or np.ndarray,
          pixel_values: Optional[np.ndarray] = None,
          labels: Optional[list or np.ndarray] = None,
          true_labels: Optional[list] = None,
          width: Optional[int] = 50,
          aspect: Optional[float] = 0.7,
          hspace: Optional[float] = 0.1,
          labelpad: Optional[float] = None,
          cmap: Optional[str or Colormap] = colors.red_transparent_blue,
          show: Optional[bool] = True):

    # support passing an explanation object

    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        # feature_names = [shap_exp.feature_names]
        # ind = 0
        if len(shap_exp.output_dims) == 1:
            shap_values = [shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])]
        elif len(shap_exp.output_dims) == 0:
            shap_values = shap_exp.values
        else:
            raise Exception("Number of outputs needs to have support added!! (probably a simple fix)")
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names
            print(shap_exp.output_names)

    # multi_output = True
    if not isinstance(shap_values, list):
        # multi_output = False
        shap_values = [shap_values]

    if len(shap_values[0].shape) == 3:
        shap_values = [v.reshape(1, *v.shape) for v in shap_values]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    # labels: (rows (images) x columns (top_k classes) )
    if labels is not None:
        if isinstance(labels, list):
            labels = np.array(labels).reshape(1, -1)

    # if labels is not None:
    #     labels = np.array(labels)
    #     if labels.shape[0] != shap_values[0].shape[0] and labels.shape[0] == len(shap_values):
    #         labels = np.tile(np.array([labels]), shap_values[0].shape[0])
    #     assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
    #     if multi_output:
    #         assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
    #     else:
    #         assert len(labels[0].shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    x = pixel_values
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    fig, axes = pl.subplots(nrows=x.shape[0], ncols=len(shap_values) + 1, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1, axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure we have a 2D array for grayscale
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])

        # if x_curr.max() > 1:
        #     x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (
                    0.2989 * x_curr[:, :, 0] + 0.5870 * x_curr[:, :, 1] + 0.1140 * x_curr[:, :, 2])  # rgb to gray
            x_curr_disp = x_curr
        elif len(x_curr.shape) == 3:
            x_curr_gray = x_curr.mean(2)

            # for non-RGB multi-channel data we show an RGB image where each of the three channels is a scaled k-mean center
            flat_vals = x_curr.reshape([x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]).T
            flat_vals = (flat_vals.T - flat_vals.mean(1)).T
            means = kmeans(flat_vals, 3, round_values=False).data.T.reshape([x_curr.shape[0], x_curr.shape[1], 3])
            x_curr_disp = (means - np.percentile(means, 0.5, (0, 1))) / (
                    np.percentile(means, 99.5, (0, 1)) - np.percentile(means, 1, (0, 1)))
            x_curr_disp[x_curr_disp > 1] = 1
            x_curr_disp[x_curr_disp < 0] = 0
        else:
            x_curr_gray = x_curr
            x_curr_disp = x_curr

        axes[row, 0].imshow(x_curr_disp, cmap=pl.get_cmap('gray'))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()

        max_val = np.nanpercentile(abs_vals, 99.9)

        for i in range(len(shap_values)):
            if labels is not None:

                axes[row, i + 1].set_title(labels[0, i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            axes[row, i + 1].imshow(x_curr_gray, cmap=pl.get_cmap('gray'), alpha=0.15,
                                    extent=(-1, sv.shape[1], sv.shape[0], -1))
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="SHAP value", orientation="horizontal",
                      aspect=fig_size[0] / aspect)
    cb.outline.set_visible(False)
    if show:
        pl.show()
        