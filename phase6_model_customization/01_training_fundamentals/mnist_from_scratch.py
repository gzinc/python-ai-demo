"""
Module: MNIST Training From Scratch - See How Models Learn

This script trains a simple neural network to recognize handwritten digits.
You'll see the complete training loop in action:
  1. Forward pass: input → prediction
  2. Loss calculation: how wrong were we?
  3. Backward pass: compute gradients (blame assignment)
  4. Optimizer step: update weights

Run with: uv run python phase6_model_customization/01_training_fundamentals/mnist_from_scratch.py
GPU mode: uv run python phase6_model_customization/01_training_fundamentals/mnist_from_scratch.py --gpu
"""

import argparse
import time
from inspect import cleandoc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# -----------------------------------------------------------------------------
# the model: a simple neural network
# -----------------------------------------------------------------------------

class SimpleNN(nn.Module):
    """
    A simple 3-layer neural network for digit classification.

    Architecture:
        Input (784) → Hidden1 (256) → Hidden2 (128) → Output (10)

    Why these numbers?
        - 784 = 28x28 pixels flattened
        - 256, 128 = arbitrary hidden sizes (could experiment)
        - 10 = digits 0-9
    """

    def __init__(self):
        super().__init__()

        # stack of layers
        self.layers = nn.Sequential(
            nn.Flatten(),              # 28x28 image → 784 vector
            nn.Linear(784, 256),       # 784 inputs → 256 outputs
            nn.ReLU(),                 # activation (non-linearity)
            nn.Linear(256, 128),       # 256 → 128
            nn.ReLU(),
            nn.Linear(128, 10),        # 128 → 10 class scores
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass: input → prediction scores"""
        return self.layers(x)


# -----------------------------------------------------------------------------
# the training loop - the heart of deep learning
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train for one epoch (one pass through all training data).

    The training loop:
        1. Forward pass: predictions = model(inputs)
        2. Loss calculation: loss = loss_fn(predictions, targets)
        3. Backward pass: loss.backward() - computes gradients
        4. Optimizer step: optimizer.step() - updates weights
    """
    model.train()  # set model to training mode
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # move data to device (CPU or GPU)
        images, labels = images.to(device), labels.to(device)

        # 1. forward pass
        predictions = model(images)

        # 2. calculate loss (how wrong are we?)
        loss = loss_fn(predictions, labels)

        # 3. backward pass (compute gradients)
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # compute new gradients

        # 4. optimizer step (update weights)
        optimizer.step()

        # track metrics
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # progress update every 100 batches
        if batch_idx % 100 == 0:
            print(f"  batch {batch_idx:4d}/{len(train_loader)} | "
                  f"loss: {loss.item():.4f} | "
                  f"acc: {100. * correct / total:.1f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """evaluate model on test set (no gradient computation)"""
    model.eval()  # set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # disable gradient computation for speed
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = loss_fn(predictions, labels)

            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# -----------------------------------------------------------------------------
# main experiment
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    print_section("MNIST Training From Scratch")

    # -------------------------------------------------------------------------
    # 1. setup device
    # -------------------------------------------------------------------------
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"using CPU {'(use --gpu for GPU)' if not args.gpu else '(CUDA not available)'}")

    # -------------------------------------------------------------------------
    # 2. load data
    # -------------------------------------------------------------------------
    print_section("Loading MNIST Dataset")

    # transforms: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    # download and load training data
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # data loaders (batch and shuffle data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    print(f"training samples: {len(train_dataset):,}")
    print(f"test samples: {len(test_dataset):,}")
    print(f"batch size: {args.batch_size}")
    print(f"batches per epoch: {len(train_loader)}")

    # -------------------------------------------------------------------------
    # 3. create model, loss function, optimizer
    # -------------------------------------------------------------------------
    print_section("Model Setup")

    model = SimpleNN().to(device)

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"model architecture:\n{model}")
    print(f"\ntotal parameters: {total_params:,}")
    print(f"trainable parameters: {trainable_params:,}")

    # loss function: cross entropy for classification
    loss_fn = nn.CrossEntropyLoss()

    # optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nloss function: CrossEntropyLoss")
    print(f"optimizer: Adam (lr={args.lr})")

    # -------------------------------------------------------------------------
    # 4. training loop
    # -------------------------------------------------------------------------
    print_section("Training")
    print(f"training for {args.epochs} epochs...\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        epoch_time = time.time() - epoch_start

        # evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(f"\n  epoch {epoch} complete in {epoch_time:.1f}s")
        print(f"  train loss: {train_loss:.4f} | train acc: {train_acc:.1f}%")
        print(f"  test loss:  {test_loss:.4f} | test acc:  {test_acc:.1f}%\n")

    total_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # 5. final results
    # -------------------------------------------------------------------------
    print_section("Training Complete")

    print(f"total training time: {total_time:.1f}s")
    print(f"final test accuracy: {test_acc:.1f}%")

    # -------------------------------------------------------------------------
    # 6. save the trained model
    # -------------------------------------------------------------------------
    print_section("Saving Trained Model")

    model_path = "mnist_model.pt"
    torch.save(model.state_dict(), model_path)

    # show file size
    import os
    file_size = os.path.getsize(model_path)
    print(f"saved to: {model_path}")
    print(f"file size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
    print(f"contains: {trainable_params:,} trained parameters")
    print(f"\nthis file IS the trained model - those 235K numbers that learned to recognize digits")

    # show what happened to the weights
    print_section("What Just Happened?")

    explanation = cleandoc("""
    You just trained a neural network! Here's what happened:

    1. INITIALIZATION
       - Created 235,146 random numbers (the weights)
       - These started as random noise - no knowledge

    2. EACH TRAINING STEP (forward → loss → backward → update)
       - Forward: fed images through the network
       - Loss: measured how wrong the predictions were
       - Backward: computed gradients (which weights caused the error)
       - Update: nudged weights in the direction that reduces error

    3. OVER TIME
       - Loss decreased: predictions got less wrong
       - Accuracy increased: more correct classifications
       - The 235K numbers now encode digit recognition knowledge

    4. THE MAGIC
       - No one told the model what a "7" looks like
       - It discovered patterns through gradient descent
       - Each weight got nudged millions of times toward the right answer

    Try experimenting:
       - Change epochs (--epochs 10)
       - Change learning rate (--lr 0.01 or --lr 0.0001)
       - Run on GPU (--gpu) to see speed difference
    """)
    print(explanation)


if __name__ == "__main__":
    main()
