from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
import base64
import io
from libs.uvit_t2i import UViT  # Import your UViT model
from libs.clip import FrozenCLIPEmbedder  # Import the CLIP embedder
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import einops
import libs.autoencoder

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the UViT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UViT(img_size=32, patch_size=2, in_chans=4, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4)
# model.load_state_dict(torch.load("models/mscoco_uvit_small.pth", map_location=device))  # Replace with your model path
model.load_state_dict(torch.load("models/mscoco_uvit_small.pth", map_location=device))
model.to(device)
model.eval()

# Load the CLIP embedder
clip_embedder = FrozenCLIPEmbedder()
clip_embedder.eval()
clip_embedder.to(device)

# Load the autoencoder
autoencoder = libs.autoencoder.get_model(pretrained_path="assets/stable-diffusion/autoencoder_kl.pth")  # Replace with your autoencoder path
autoencoder.to(device)

# Define the DPM Solver sampling function
def dpm_solver_sample(n_samples, sample_steps, context, z_shape, betas):
    _z_init = torch.randn(n_samples, *z_shape, device=device)  # Initialize latent space
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * len(betas)  # Scale timesteps
        return model(x, t, context=context)

    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    _z = dpm_solver.sample(_z_init, steps=sample_steps, eps=1. / len(betas), T=1.)
    return autoencoder.decode(_z)  # Decode latent space into an image

@app.route('/')
def index():
    return render_template('index.html')  # HTML frontend

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get the text prompt from the request
        data = request.get_json()
        prompt = data.get('prompt', None)
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Encode the text prompt into a latent feature using CLIP
        with torch.no_grad():
            context = clip_embedder.encode([prompt]).to(device)  # Encode the prompt into a latent feature
            print(f"Context shape: {context.shape}")  # Debugging statement

        # Define DPM Solver parameters
        n_samples = 1  # Number of images to generate
        sample_steps = 50  # Number of sampling steps
        z_shape = (4, 32, 32)  # Latent space shape (adjust based on your model)
        betas = torch.linspace(0.00085, 0.0120, 1000).numpy()  # Beta schedule

        # Generate an image using the DPM Solver
        with torch.no_grad():
            generated_images = dpm_solver_sample(n_samples, sample_steps, context, z_shape, betas)
            print(f"Generated images shape: {generated_images.shape}")  # Debugging statement

        # Convert the generated image to a base64 string
        image = generated_images[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        image = (image * 255).clip(0, 255).astype("uint8")  # Scale to [0, 255]
        pil_image = Image.fromarray(image)

        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'image': image_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)