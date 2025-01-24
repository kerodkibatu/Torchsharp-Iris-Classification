# Iris Classification with TorchSharp

This project demonstrates how to build and train a simple feedforward neural network using TorchSharp in C# to classify the Iris dataset. The model is trained to predict the species of Iris flowers based on four features: sepal length, sepal width, petal length, and petal width.

## Features
- Load and preprocess the Iris dataset from a CSV file.
- Normalize features using training set statistics.
- Train a two-layer neural network with ReLU activation.
- Evaluate model performance on a test set.
- Save the trained model to a file.

## Requirements
- [.NET SDK](https://dotnet.microsoft.com/download)
- [TorchSharp](https://github.com/dotnet/TorchSharp) (install via NuGet)

## Dataset
The Iris dataset is included in the `Iris.csv` file. It contains 150 samples with four features and three classes:
- **Features**: Sepal length, sepal width, petal length, petal width.
- **Classes**: Iris-setosa, Iris-versicolor, Iris-virginica.

## Code Overview
1. **Data Loading**: The dataset is loaded from `Iris.csv` and split into training and test sets.
2. **Preprocessing**: Features are normalized using the mean and standard deviation of the training set.
3. **Model Definition**: A simple feedforward neural network with one hidden layer and ReLU activation is defined.
4. **Training**: The model is trained using the Adam optimizer and cross-entropy loss.
5. **Evaluation**: The model's accuracy is evaluated on the test set.
6. **Saving**: The trained model is saved to a `.pt` file.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/kerodkibatu/TorchSharp-Iris.git
   cd TorchSharp-Iris
   ```
2. Install dependencies:
   ```bash
   dotnet restore
   ```
3. Run the project:
   ```bash
   dotnet run
   ```

## Example Output
```
Epoch [20/200], Loss: 0.8765; Train Accuracy: 85.00%
Epoch [40/200], Loss: 0.6543; Train Accuracy: 90.00%
...
Epoch [200/200], Loss: 0.1234; Train Accuracy: 98.33%
Test Accuracy: 96.67%
```

## Saving the Model
The trained model is saved to a file with a timestamp, e.g., `iris_model_20250124205215.pt`.

## What's Next?
- Experiment with different neural network architectures.
- Try different optimizers and learning rates.
- Explore other datasets and classification tasks.
- Build a simple API to serve the model predictions.
- Build a simple web application to interact with the model.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE.txt) file for details.