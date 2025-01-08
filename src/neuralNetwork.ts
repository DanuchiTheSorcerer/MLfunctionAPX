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
        new Matrix(layerSizes[i + 1], layerSizes[i], () =>
          Math.random()
        )
      ); // Random initialization

      // Create a vector for the biases of layer i+1
      this.biases.push(new Vector(layerSizes[i + 1], () => 0)); // Random initialization
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
      (i:number) => 1 / (1 + Math.exp(-input.components[i]))
    );
  }

  private sigmoidPrime(input:Vector): Vector {
    let sigm = this.sigmoid(input)
    let oneVec = new Vector(input.components.length,()=>1)
    return sigm.mult(oneVec.subtract(sigm))
  }

  public train(input: Vector, target: Vector, learningRate: number): void {
    // Forward pass
    let activations: Vector[] = [];
    let zActivations: Vector[] = [];
  
    activations[0] = input; // Input is the first activation
    for (let j = 1; j < this.layerSizes.length; j++) {
      const z = activations[j - 1].transform(this.weights[j - 1]).add(this.biases[j - 1]); // z = W * a + b
      zActivations[j] = z; // Store pre-activation values
      activations[j] = this.sigmoid(z); // Apply activation function
    }
  
    // Backward pass
    let weightGradients: Matrix[] = [];
    let biasGradients: Vector[] = [];
    let delta: Vector;
  
    // Compute output layer error
    const outputError = activations[this.layerSizes.length - 1].subtract(target); // Error = output - target
    delta = outputError.mult(this.sigmoidPrime(zActivations[this.layerSizes.length - 1])); // Delta = error * sigmoid'(z)
  
    // Backpropagate
    for (let j = this.layerSizes.length - 2; j >= 0; j--) {
      // Gradients for weights: delta * activations[j]^T
      weightGradients[j] = new Matrix(
        this.layerSizes[j + 1],
        this.layerSizes[j],
        (row, col) => delta.components[row] * activations[j].components[col]
      );
  
      // Gradients for biases: delta
      biasGradients[j] = delta;
  
      // Compute delta for the previous layer
      if (j > 0) {
        delta = delta.transform(this.weights[j].transpose()).mult(this.sigmoidPrime(zActivations[j]))
    
      }
    }  
    // Update weights and biases
    for (let j = 0; j < this.weights.length; j++) {
      this.weights[j] = this.weights[j].subtract(weightGradients[j].scale(learningRate)); // Update weights
      this.biases[j] = this.biases[j].subtract(biasGradients[j].scale(learningRate)); // Update biases
    }
  }
}
