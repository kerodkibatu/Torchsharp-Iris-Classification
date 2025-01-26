using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// Set seed for reproducibility
torch.manual_seed(42);

// Load Iris dataset and skip header
var data = File.ReadAllLines("Iris.csv")[1..];

string[] classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"];

var featuresRaw = new float[150, 4];
var labelsRaw = new int[150];

for (int i = 0; i < 150; i++)
{
    var row = data[i].Split(',');
    for (int j = 0; j < 4; j++)
    {
        featuresRaw[i, j] = float.Parse(row[j]);
    }

    labelsRaw[i] = Array.IndexOf(classes, row[5]);
}

// Convert data to tensors
var features = torch.tensor(featuresRaw, dtype: ScalarType.Float32);
var labels = torch.tensor(labelsRaw, dtype: ScalarType.Int64);

// Split dataset into training and test sets; 80% training, 20% test split
int trainSize = (int)(0.8 * features.shape[0]);
var indices = torch.randperm(features.shape[0]);
var trainFeatures = features.index(indices[..trainSize]);
var trainLabels = labels.index(indices[..trainSize]);
var testFeatures = features.index(indices[trainSize..]);
var testLabels = labels.index(indices[trainSize..]);

// Normalize features using training statistics
var mean = trainFeatures.mean([0]);
var std = trainFeatures.std(0);
trainFeatures = (trainFeatures - mean) / std;
testFeatures = (testFeatures - mean) / std;

// Hyperparameters
int inputDim = 4;
int hiddenDim = 3;
int outputDim = 3;
int numEpochs = 200;
float learningRate = 0.01f;

// Create model, loss function, and optimizer
using var model = new IrisModel(inputDim, hiddenDim, outputDim);
var criterion = CrossEntropyLoss();
var optimizer = torch.optim.Adam(model.parameters(), learningRate);

// Training loop
for (int epoch = 0; epoch < numEpochs; epoch++)
{
    // Forward pass
    using var outputs = model.forward(trainFeatures);
    using var loss = criterion.forward(outputs, trainLabels);

    // Backward pass and optimize
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // Test accuracy
    using (torch.no_grad())
    {
        var predicted = outputs.argmax(1);
        var correct = (predicted == trainLabels).sum().ToInt32();
        var accuracy = correct / (float)trainLabels.shape[0];

        // Print progress every 20 epochs
        if ((epoch + 1) % 20 == 0)
        {
            Console.WriteLine($"Epoch [{epoch + 1}/{numEpochs}], Loss: {loss.ToSingle():F4}; Train Accuracy: {accuracy * 100:F2}%");
        }
    }
}

// Evaluation
using (torch.no_grad())
{
    var testOutputs = model.forward(testFeatures);
    var predicted = testOutputs.argmax(1);
    var correct = (predicted == testLabels).sum().ToInt32();
    var accuracy = correct / (float)testLabels.shape[0];
    Console.WriteLine($"Test Accuracy: {accuracy * 100:F2}%");
}

// Save model
model.save($"iris_model_{DateTime.Now:yyyyMMddHHmmss}.pt");

// Cleanup
features.Dispose();
labels.Dispose();
trainFeatures.Dispose();
trainLabels.Dispose();
testFeatures.Dispose();
testLabels.Dispose();

// Simple Feedforward Neural Network for Classification
public class IrisModel : Module<Tensor, Tensor>
{
    // Linear layers
    private Linear layer1;
    private Linear layer2;
    // ReLU activation function
    private ReLU relu;

    public IrisModel(int inputDim, int hiddenDim, int outputDim) : base("IrisModel")
    {
        layer1 = Linear(inputDim, hiddenDim);
        layer2 = Linear(hiddenDim, outputDim);
        relu = ReLU();

        RegisterComponents();
    }

    public override Tensor forward(Tensor input)
    {
        var x = layer1.forward(input);
        x = relu.forward(x);
        x = layer2.forward(x);
        return x;
    }
}