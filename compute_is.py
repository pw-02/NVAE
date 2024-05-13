import torch
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import entropy
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image

# Load pre-trained Inception model
def load_inception_model():
    model = inception_v3(pretrained=True)
    model.eval()  # Set to evaluation mode
    return model

# Function to get Inception probabilities from a DataLoader
def get_inception_probs(data_loader, model):
    all_probs = []

    with torch.no_grad():  # No gradient calculation needed
        for batch in data_loader:
            images = batch[0]  # Extract the images from the batch
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # Get probabilities
            all_probs.append(probs.numpy())  # Store probabilities as numpy arrays
    
    return np.vstack(all_probs)  # Concatenate all probabilities into a single array

# Function to calculate the Inception Score
def calculate_inception_score(data_loader, model, num_splits=10):
    # Get probabilities for all images in the DataLoader
    probs = get_inception_probs(data_loader, model)

    # Calculate the marginal probability
    p_y = np.mean(probs, axis=0)

    scores = []
    # Calculate Inception Score for each split
    for split in np.array_split(probs, num_splits):
        # Calculate KL divergence between each sample and the marginal distribution
        kl_divergence = [entropy(p, p_y) for p in split]
        # Compute the exponential of the mean KL divergence
        scores.append(np.exp(np.mean(kl_divergence)))

    # Return the mean and standard deviation of the Inception Score
    return np.mean(scores), np.std(scores)

# Argument parser for command-line arguments
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--image-path', type=str, help='Path to the folder containing generated images', default='is_images')
parser.add_argument('--num-splits', type=int, default=10, help='Number of splits for IS calculation')
parser.add_argument('--batch-size', type=int, default=50, help='Batch size for data loading')

args = parser.parse_args()

def main():
    # Load the pre-trained Inception model
    model = load_inception_model()

    # Apply a transformation to convert images to tensors
    dataset_transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Ensure correct size for Inception
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
    ])
    
    # Load images from the specified folder with the required transformation
    image_dataset = ImageFolder(args.image_path, transform=dataset_transform)  # Load dataset with transformation
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False)  # DataLoader for batch processing
    
    # Calculate the Inception Score with the DataLoader
    inception_mean, inception_std = calculate_inception_score(image_loader, model, num_splits=args.num_splits)
    
    print(f"Inception Score: {inception_mean} ± {inception_std}")

if __name__ == '__main__':
    main()
