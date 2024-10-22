# LeNet Project

This project implements the LeNet convolutional neural network architecture using PyTorch. The LeNet model is designed to perform image classification, particularly on datasets like MNIST.

## Project Overview

- **LeNet Architecture**: The project implements the classic LeNet-5 architecture, which is widely used for handwritten digit classification.
- **PyTorch Implementation**: The model is built and trained using PyTorch for efficient computation and GPU support.
- **Data**: Although no dataset is provided, this project is designed to work with standard image classification datasets such as MNIST.

## Project Structure

- **LeNet.py**: The main Python script that implements the LeNet model and handles training and evaluation.
- **best/**: Directory where the best performing models are saved during training.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/LeNet.git
    cd LeNet
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Train the LeNet Model**:
    Execute the training script to train the LeNet model:
    ```bash
    python LeNet.py
    ```

2. **Evaluate the Model**:
    The script will output the evaluation results of the model on a test dataset.

## Project Workflow

1. **Data Loading**: Load the dataset (e.g., MNIST) for training the LeNet model.
2. **Model Training**: Train the LeNet model on the dataset and save the best performing model.
3. **Evaluation**: Evaluate the trained model on a test set and view the results.
