def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # create a model that maps the input image to the activations
    # of the last conv layer and the output predictions

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )


    # compute the gradient of the top predicted class for input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array 
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    #heatmap = 1 - heatmap
    return heatmap.numpy()

import matplotlib
def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Load the original image

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    #jet = cm.get_cmap("jet")
    jet = matplotlib.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Invert the colormap
    #jet_heatmap = 1 - jet_heatmap  # Invert the heatmap values

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    #superimposed_img.save(cam_path) , cam_path="cam.jpg"

    # Display Grad CAM
    #plt.matshow(superimposed_img)
    #plt.show()
    return superimposed_img

#put name of the last conv layer
last_conv_layer_name = 'conv2d_2'
model.layers[-1].activation = None

plt.figure(figsize=(10, 8))  
num_images = 4
indices = np.random.choice(len(test_data), num_images, replace=False)
preds = model.predict(train_data)

for i, idx in enumerate(indices):
    # Display original image
    pred_label = preds[i]
    plt.subplot(2, num_images, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_data[idx], cmap=plt.cm.binary)

    # Display GradCAM heatmap
    plt.subplot(2, num_images, num_images + i + 1)
    img = test_data[idx]
    img_array = np.expand_dims(img, axis=0)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    overlaid_img = save_and_display_gradcam(img, heatmap)
    plt.title(f'Original Image\nPrediction: {pred_label}')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(overlaid_img, cmap='jet')

# Adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()
