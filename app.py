import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import vgg19
from torchvision.utils import save_image
import torch.optim as optim
import torch.nn as nn

# Title
st.title("Neural Style Transfer")

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load and transform image
def load_image(image_file):
    image = Image.open(image_file)
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# Function to show tensor as image
def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)
    image = image * 255
    image = image.clip(0, 255).astype("uint8")
    return Image.fromarray(image)

# Uploading images
content_image_file = st.file_uploader("Upload Content Image", type=["jpg", "png"])
style_image_file = st.file_uploader("Upload Style Image", type=["jpg", "png"])

if content_image_file:
    st.write("Content Image Uploaded Successfully!")
    content_image = load_image(content_image_file)
    st.image(im_convert(content_image), caption="Content Image")

if style_image_file:
    st.write("Style Image Uploaded Successfully!")
    style_image = load_image(style_image_file)
    st.image(im_convert(style_image), caption="Style Image")

if content_image_file is not None and style_image_file is not None:
    # Load VGG19 model
    vgg = vgg19(pretrained=True).features.to(device).eval()

    # Style transfer procedure

    # Content and style targets
    content_target = content_image.clone().requires_grad_(False).to(device)
    style_target = style_image.clone().requires_grad_(False).to(device)

    # Generated image
    generated = content_image.clone().requires_grad_(True).to(device)

    # Define optimizer
    optimizer = optim.Adam([generated], lr=0.003)

    # Define loss functions
    mse_loss = nn.MSELoss()

    # Content layers and style layers
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_features(image, model):
        layers = {
            '0': 'conv_1', '5': 'conv_2', '10': 'conv_3',
            '19': 'conv_4', '28': 'conv_5'
        }
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    # Compute style loss
    def compute_gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram_matrix = torch.mm(tensor, tensor.t())
        return gram_matrix

    # Extract features
    content_features = get_features(content_image, vgg)
    style_features = get_features(style_image, vgg)

    style_grams = {layer: compute_gram_matrix(style_features[layer]) for layer in style_features}

    # Run style transfer
    steps = 5000
    style_weight = 1e6
    content_weight = 1e4

    for step in range(steps):
        generated_features = get_features(generated, vgg)

        content_loss = mse_loss(generated_features['conv_4'], content_features['conv_4'])

        style_loss = 0
        for layer in style_grams:
            generated_gram = compute_gram_matrix(generated_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = mse_loss(generated_gram, style_gram)
            _, d, h, w = generated_features[layer].shape
            style_loss += layer_style_loss / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)  # Retain the graph
        optimizer.step()

        # Display the image at every 1000 steps
        if step % 1000 == 0:
            st.write(f"Step {step}/{steps}: Loss: {total_loss.item()}")
            st.image(im_convert(generated), caption=f"Generated Image at Step {step}")

    # Final Output
    st.image(im_convert(generated), caption="Final Generated Image")
