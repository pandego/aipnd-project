import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def test_model(model, dataloaders, device):
    model.eval()
    test_loss = 0
    accuracy = 0
    criterion = nn.NLLLoss()

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss / len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {accuracy / len(dataloaders['test']):.3f}")


def main():
    # Argument Parsing
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('data_directory', type=str, help='Data directory for training and validation')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['vgg16', 'vgg13', 'densenet121', 'alexnet', 'resnet18'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units for the classifier')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()

    # Data loading and transformation (similar to notebook code)
    # Define data directories
    data_dir = args.data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # Using the image datasets and the transforms, define the dataloaders
    batch_size = 8
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size)
    }

    # Model setup and training (taking into account the arguments provided)
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif args.arch == "densenet121":
        model = models.densenet121(pretrained=True)
        input_size = 1024
    elif args.arch == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif args.arch == "resnet18":  # adding support for resnet18
        model = models.resnet18(pretrained=True)
        input_size = 512
    else:
        print("Model architecture not recognized. Using ResNet18 as default.")
        model = models.resnet18(pretrained=True)
        input_size = 512

    # Freeze the feature parameters, so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Define our classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),

        nn.Linear(args.hidden_units, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.4),

        nn.Linear(512, len(cat_to_name)),
        nn.LogSoftmax(dim=1)
    )

    # Attach the classifier to the right attribute of the model
    if args.arch in ['vgg16', 'vgg13', 'densenet121', 'alexnet']:
        model.classifier = classifier
    elif args.arch in ['resnet18']:
        model.fc = classifier
    else:
        # Default to resnet18
        model.fc = classifier

    # Define criterion and optimizer
    if args.arch in ['vgg16', 'vgg13', 'densenet121', 'alexnet']:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    elif args.arch == 'resnet18':
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        print(f"Architecture {args.arch} not recognized.")
        exit()

    criterion = nn.NLLLoss()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using {device} for training...")
    model.to(device)

    # Train the classifier
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 2

    print(f"Start training on architecture {args.arch}...")

    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Validation loop
            if steps % print_every == 0:
                model.eval()
                val_loss = 0
                accuracy = 0

                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    val_loss += loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {val_loss / len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

        # Freeing GPU memory after validation
        torch.cuda.empty_cache()

    print("Training finished!")
    print("###")

    # Prints out training loss, validation loss, and validation accuracy as the network trains
    test_model(model, dataloaders, device)
    print("###")

    # Saving checkpoint (to args.save_dir if provided)
    print("Saving model...")
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Saveing the checkpoint
    checkpoint = {
        'input_size': input_size,
        'output_size': len(cat_to_name),
        'arch': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")
    print(f"Model saved: '{args.save_dir}/checkpoint.pth'")


if __name__ == '__main__':
    main()
