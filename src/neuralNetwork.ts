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
      // Xavier Initialization: Random weights scaled by sqrt(1 / number of inputs)
      const fanIn = layerSizes[i]; // Number of inputs to the layer
      this.weights.push(
        new Matrix(layerSizes[i + 1], layerSizes[i], () =>
          Math.random() * Math.sqrt(2 / fanIn) * (Math.random() > 0.5 ? 1 : -1) // Ensures weights are properly scaled
        )
      );
    
      // Create a vector for the biases of layer i+1 (start at zero for simplicity)
      this.biases.push(new Vector(layerSizes[i + 1], () => 0));
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
    // Implementing tanh instead of sigmoid
    return new Vector(
      input.components.length,
      (i: number) => Math.tanh(input.components[i]) // tanh function
    );
}

private sigmoidPrime(input: Vector): Vector {
    // Derivative of tanh: 1 - tanh^2(x)
    let tanhVec = this.sigmoid(input); // Using tanh implementation
    let oneVec = new Vector(input.components.length, () => 1);
    return oneVec.subtract(tanhVec.mult(tanhVec)); // 1 - tanh^2(x)
}

  public train(inputs: Vector[], targets: Vector[], learningRate: number): void {
    const batchSize = inputs.length;
    const weightGradients: Matrix[] = this.weights.map(
        w => new Matrix(w.rows, w.columns, () => 0)
    );
    const biasGradients: Vector[] = this.biases.map(
        b => new Vector(b.components.length, () => 0)
    );

    // Accumulate gradients over the batch
    for (let i = 0; i < batchSize; i++) {
        const input = inputs[i];
        const target = targets[i];

        // Forward pass
        const activations: Vector[] = [];
        const zActivations: Vector[] = [];

        activations[0] = input;
        for (let j = 1; j < this.layerSizes.length; j++) {
            const z = activations[j - 1].transform(this.weights[j - 1]).add(this.biases[j - 1]);
            zActivations[j] = z;
            activations[j] = this.sigmoid(z); // Or use tanh for better results
        }

        // Backward pass
        let delta = activations[this.layerSizes.length - 1]
            .subtract(target)
            .mult(this.sigmoidPrime(zActivations[this.layerSizes.length - 1]));

        for (let j = this.layerSizes.length - 2; j >= 0; j--) {
            // Update weight gradients
            const weightGradient = new Matrix(
                this.layerSizes[j + 1],
                this.layerSizes[j],
                (row, col) => delta.components[row] * activations[j].components[col]
            );
            weightGradients[j] = weightGradients[j].add(weightGradient);

            // Update bias gradients
            biasGradients[j] = biasGradients[j].add(delta);

            // Backpropagate delta
            if (j > 0) {
                delta = delta.transform(this.weights[j].transpose())
                    .mult(this.sigmoidPrime(zActivations[j]));
            }
        }
    }

    // Average gradients and update weights and biases
    for (let j = 0; j < this.weights.length; j++) {
        weightGradients[j] = weightGradients[j].scale(1 / batchSize);
        biasGradients[j] = biasGradients[j].scale(1 / batchSize);

        this.weights[j] = this.weights[j].subtract(weightGradients[j].scale(learningRate));
        this.biases[j] = this.biases[j].subtract(biasGradients[j].scale(learningRate));
    }
}
}
