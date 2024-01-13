import cv2
import torch

conv_shallow_params = {
    'num_slices': 50,
    'slice_shape': (3, 240, 20),
    'latent_size': 128,
    'model_file': 'out/model_conv_shallow_decoder.pt'
}

conv_deep_params = {
    'num_slices': 31,
    'slice_shape': (3, 240, 32),
    'latent_size': 256,
    'model_file': 'out/model_conv_deep_decoder.pt'
}

parms = conv_deep_params

DEVICE = 'mps'

model = torch.jit.load(parms['model_file'])
model.to(DEVICE)


class LatentSpaceExplorer:
    def __init__(self, latent_dim, device, num_steps=10):
        self.latent_dim = latent_dim
        self.device = device
        self.num_steps = num_steps
        self.current_step = 0
        self.point_a, self.point_b = self._generate_random_point(), self._generate_random_point()

    def _generate_random_point(self):
        # Generates two random points in the latent space
        return torch.randn(self.latent_dim, device=self.device)

    def _linear_interpolate(self, t):
        # Linear interpolation between point_a and point_b at t
        return (1 - t) * self.point_a + t * self.point_b

    def get_next_point(self):
        if self.current_step >= self.num_steps:
            # Start the new line from the endpoint of the last one.
            # This makes a smoother transition
            self.point_a = self.point_b
            self.point_b = self._generate_random_point()
            self.current_step = 0

        # Calculate interpolation factor
        t = self.current_step / float(self.num_steps - 1)
        point = self._linear_interpolate(t)
        self.current_step += 1
        return point


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


def sample_vae(model, lse, num_slices):
    with torch.no_grad():
        slices = []
        for i in range(num_slices):
            slices.append(lse.get_next_point())
        noise = torch.stack(slices)
        generated_images = model(noise)
        return generated_images


def generate_and_display_images(sample_vae, model):
    window_name = 'VAE Landscape Movie'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    lse = LatentSpaceExplorer(parms['latent_size'], DEVICE, parms['num_slices'])

    delay = 50

    try:
        while True:
            # Sample images from your VAE (adjust the function call as necessary)
            x = sample_vae(model, lse, parms['num_slices'])

            # Use your existing function to stitch images
            stitched_image = stitch_images(x, parms['slice_shape'])

            # Convert the Matplotlib figure to an OpenCV image
            stitched_image_cv = stitched_image[:, :, ::-1].copy()  # Convert RGB to BGR

            # Display the image
            cv2.imshow(window_name, stitched_image_cv)

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                delay = 1000 if delay == 50 else 50
    finally:
        cv2.destroyAllWindows()


# Call the function to start the display
generate_and_display_images(sample_vae, model)
