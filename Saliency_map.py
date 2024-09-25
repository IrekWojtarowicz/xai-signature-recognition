! pip install tf-keras-vis
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def show_saliency_map(image, model, image_id) :
    img_batch = (np.expand_dims(image, 0))
    # creating saliency object
    saliency = Saliency(model)
    # creating loss function
    loss = lambda output: tf.keras.backend.mean(output[:, tf.argmax(train_labels_c[image_id])])

    # creating and normalizing saliency map
    saliency_map = saliency(loss, img_batch)
    saliency_map = normalize(saliency_map)

    # reshaping for vizualization
    sal_vis = saliency_map.reshape(saliency_map.shape[1], saliency_map.shape[2])

    # showing the map
    return sal_vis

import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(test_data)
predictions_class = [np.argmax(predictions[i]) for i in random_indices_list]


plt.figure(figsize=(10, 8))
num_images = len(train_subset)

for i in range(num_images):
    # Display the original image with predicted and true labels
    plt.subplot(2, num_images, i + 1)  # First row for images
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_subset[i])  # Plot the original image

    # Add titles for predicted and true labels
    true_label = train_labels[random_indices_list[i]]   # Get the true label for the current index
    predicted_label = predictions_class[i]  # Get the predicted label for the current index
    plt.title(f"Pred: {predicted_label}\nTrue: {true_label}", fontsize=14, pad=20)  # Set title with padding

    # Display the saliency map
    plt.subplot(2, num_images, num_images + i + 1)  # Second row for saliency maps
    saliency_map = show_saliency_map((train_subset)[i], model, i)  # Get the saliency map
    plt.imshow(saliency_map, cmap='viridis')  # Plot the saliency map
    plt.axis('off')  # Hide axis

plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Adjust spacing between subplots 
plt.tight_layout()
plt.show()
