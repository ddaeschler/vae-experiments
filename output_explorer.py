import cv2
import torch
import numpy as np
import joblib

conv_shallow_params = {
    'num_slices': 50,
    'slice_shape': (3, 240, 20),
    'latent_size': 128,
    'model_file': 'out/model_conv_shallow.pt'
}

conv_deep_params = {
    'num_slices': 31,
    'slice_shape': (3, 240, 32),
    'latent_size': 384,
    'model_file': 'out/model_conv_deep.pt'
}

parms = conv_shallow_params

DEVICE = 'mps'

model = torch.jit.load(parms['model_file'])
model.to(DEVICE)

def stitch_images(x, slice_shape):
    """
    Stitch image slices horizontally and return the resulting single image as a NumPy array.

    :param x: Tensor containing the image slices.
    :param slice_shape: Shape of each slice (channels, height, width).
    """
    x = x.view(-1, *slice_shape)  # Reshape
    x = x.permute(0, 2, 3, 1)  # Reorder dimensions for plotting

    # Concatenate all slices horizontally
    stitched_image = torch.cat(tuple(x), dim=1)

    # Convert to numpy
    stitched_image_np = stitched_image.cpu().numpy()

    return stitched_image_np


def sample_vae(model, num_slices, latent_size):
    with torch.no_grad():
        noise = torch.randn(num_slices, latent_size).to(DEVICE)
        generated_images = model(noise)
        return generated_images


def generate_and_display_images(sample_vae, model):
    window_name = 'VAE Landscape Movie'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # Sample images from your VAE (adjust the function call as necessary)
            x = sample_vae(model, parms['num_slices'], parms['latent_size'])

            # Use your existing function to stitch images
            stitched_image = stitch_images(x, parms['slice_shape'])

            # Convert the Matplotlib figure to an OpenCV image
            stitched_image_cv = np.array(stitched_image)
            stitched_image_cv = stitched_image_cv[:, :, ::-1].copy()  # Convert RGB to BGR

            # Display the image
            cv2.imshow(window_name, stitched_image_cv)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()


# Call the function to start the display
generate_and_display_images(sample_vae, model)
