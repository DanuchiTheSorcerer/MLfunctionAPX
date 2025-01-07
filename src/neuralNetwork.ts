import { Matrix, Vector } from './linearAlgebra.ts';

export class NeuralNetwork {
  // 1. Array of weights (matrices)
  public weights: Matrix[];

  // 2. Array of biases (vectors)
  public biases: Vector[];

  // 3. Array of layer sizes (numbers)
  public layerSizes: number[];

  // Constructor initializes the neural network with layer sizes and creates the weights and biases
  public constructor(layerSizes: number[]) {
    this.layerSizes = layerSizes;

    // Create weights and biases for each layer
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < layerSizes.length - 1; i++) {
      // Create a matrix for the weights between layer i and layer i+1
      this.weights.push(
        new Matrix(layerSizes[i + 1], layerSizes[i], (row, col) =>
          Math.random()
        )
      ); // Random initialization

      // Create a vector for the biases of layer i+1
      this.biases.push(new Vector(layerSizes[i + 1], () => Math.random())); // Random initialization
    }
  }

  // 4. Forward method that takes in an input vector and outputs a result vector
  public forwards(input: Vector): Vector {
    let activations = input; // Start with the input vector

    // Propagate through the layers
    for (let i = 0; i < this.weights.length; i++) {
      // Apply the weight matrix and add the bias vector
      // Use the `transform` method to apply the weight matrix (Matrix -> Vector)
      activations = activations.transform(this.weights[i]).add(this.biases[i]);
      // Apply the sigmoid activation function
      activations = this.sigmoid(activations);
    }

    return activations;
  }

  // 5. Private sigmoid function to apply sigmoid activation function element-wise
  private sigmoid(input: Vector): Vector {
    return new Vector(
      input.components.length,
      (i) => 1 / (1 + Math.exp(-input.components[i]))
    );
  }
}
