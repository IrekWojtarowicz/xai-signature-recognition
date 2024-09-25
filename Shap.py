!pip install shap
import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pyplot as pl
from shap._explanation import Explanation
from shap.utils import ordinal_str
from shap.utils._legacy import kmeans
from shap.plots import colors

name_list = ["genuine", "forged"]
train_data_subset = train_subset_forshap
# Define an Image masker with "inpaint_telea" method and shape of the first image in train_data_subset
masker = shap.maskers.Image("inpaint_telea", train_data_subset[0].shape)

# Initialize an Explainer using PartitionExplainer with the Image masker
explainer = shap.Explainer(model, masker, output_names=name_list, base_values = 0.5)

# Generate SHAP values for the subset
shap_values = explainer(train_data_subset, max_evals=250, batch_size=50)
print(shap_values.base_values)

plot_shap(shap_values, labels = name_list, show = False, true_labels=predictions_class)
plt.show()
