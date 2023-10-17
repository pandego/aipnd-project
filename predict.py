import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint['arch']

    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Architecture {arch} not recognized")

    for param in model.parameters():
        param.requires_grad = False

    if arch in ['vgg16', 'vgg13', 'densenet121', 'alexnet']:
        model.classifier = checkpoint['classifier']
    elif arch in ['resnet18']:
        model.fc = checkpoint['classifier']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # optimizer = None
    # epochs = checkpoint.get('epochs', 0)

    return model  # , optimizer, epochs


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    # Open the image
    img = Image.open(image_path)

    # Resize the image
    width, height = img.size
    aspect_ratio = width / height
    if width < height:
        new_width = 256
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 256
        new_width = int(new_height * aspect_ratio)
    img = img.resize((new_width, new_height))

    # Center crop to 224x224
    left_margin = (new_width - 224) / 2
    bottom_margin = (new_height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Convert color channels to 0-1
    np_img = np.array(img) / 255

    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std

    # Reorder dimensions
    np_img = np_img.transpose((2, 0, 1))

    # Convert to a PyTorch tensor
    return torch.from_numpy(np_img)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def predict(image_path, model, use_gpu, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using {device} for prediction...")
    model.to(device)
    model.eval()

    # Process the image
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.float()

    # Move image tensor to the device
    img_tensor = img_tensor.to(device)

    # Predict the class
    with torch.no_grad():
        output = model.forward(img_tensor)

    # Calculate the class probabilities
    probabilities = torch.exp(output)

    # Get the top-k probabilities and indices
    top_probs, top_indices = probabilities.topk(top_k)

    # Convert top_probs and top_indices to lists
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_indices = top_indices.detach().cpu().numpy().tolist()[0]

    # Convert top_indices to actual class labels
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes


def display_prediction(image_path, model, cat_to_name, use_gpu, top_k):
    ''' Display image and predictions from model
    '''

    # Predict the top k classes
    probs, classes = predict(image_path, model, use_gpu, top_k)
    class_names = [cat_to_name[str(cls)] for cls in classes]

    # Console output for the predicted flower name and its probability
    print(f"Predicted Flower Name: {class_names[0]}")
    print(f"Probability: {probs[0]:.2f}")

    # Display the image
    plt.figure(figsize=(6, 10))
    ax = plt.subplot(2, 1, 1)
    img = process_image(image_path)
    imshow(img, ax, title=class_names[0])  # The most probable class as title

    # Display the top 5 classes' probabilities
    plt.subplot(2, 1, 2)
    plt.barh(class_names, probs, color='blue')
    plt.xlabel('Probability')
    plt.ylabel('Class')
    plt.gca().invert_yaxis()  # To display the highest probability at the top

    # Save the plot
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{image_path.split('/')[-1].split('.')[0]}_prediction_output.png"
    plt.savefig(output_file)
    print(f"Prediction plot saved to '{output_file}'")
    # plt.show()  # Uncomment if you still want to display the plot


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to image for prediction')
    parser.add_argument('checkpoint', type=str, help='Checkpoint for model loading')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='', help='JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()

    # Load checkpoint and model
    # model, optimizer, epochs = load_checkpoint(args.checkpoint)
    model = load_checkpoint(args.checkpoint)

    # Process image
    image = process_image(args.image_path)

    # Use category_names if provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

    # Prediction and displaying results
    display_prediction(args.image_path, model, cat_to_name, args.gpu, args.top_k)


if __name__ == '__main__':
    main()
