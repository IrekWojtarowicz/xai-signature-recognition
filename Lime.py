# Install the LIME package
!pip install lime
import os
import lime
from lime import lime_tabular
from lime.lime_image import LimeImageExplainer

predictions = model.predict(test_data)
predictions_class = [np.argmax(predictions[i]) for i in random_indices_list]

def predict_fn(images):
    return model.predict(images)

from skimage.segmentation import mark_boundaries
plt.figure(figsize=(15, 10))
num_images = len(train_subset)
explainer = LimeImageExplainer()

for i in range(num_images):
    img = train_subset[i]
    pred_label = predictions_class[i]
    # Explain the prediction for the image using Lime
    explanation = explainer.explain_instance(img, predict_fn, top_labels=1, hide_color=0)
    explanation_image, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)


    img = train_subset[i]
    pred_label = predictions_class[i]
   # Plot original image with predicted label
    plt.subplot(num_images, 2, 2*i + 1)
    plt.imshow(img)
    plt.axis('off')

    true_label = train_labels[random_indices_list[i]]  
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=14, pad=20)  

    # Plot Lime explanation on the right
    plt.subplot(num_images, 2, 2*i + 2)
    plt.imshow(mark_boundaries(explanation_image/ 2 + 0.5, mask))
    plt.title('LIME Explanation')
    plt.axis('off')

plt.tight_layout()
plt.show()
