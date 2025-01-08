export class Vector {
  public components: number[] = [];
  public constructor(components: number, genFunc: Function) {
    this.components = [];
    for (let i = 0; i < components; i++) {
      this.components[i] = genFunc(i);
    }
  }
  public add(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = otherVector.components[i] + this.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public mult(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = otherVector.components[i] * this.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public scale(scalar: number): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = this.components[i]*scalar;
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public subtract(otherVector: Vector): Vector {
    let newComponents: number[] = [];
    for (let i = 0; i < this.components.length; i++) {
      newComponents[i] = this.components[i] - otherVector.components[i];
    }

    return new Vector(this.components.length, (i: number) => newComponents[i]);
  }
  public transform(transformer: Matrix): Vector {
    if (transformer.columns !== this.components.length) {
      throw new Error(
        'Matrix columns must match the number of vector components.'
      );
    }

    let newComponents: number[] = [];

    for (let i = 0; i < transformer.rows; i++) {
      let sum = 0;
      for (let j = 0; j < transformer.columns; j++) {
        sum += transformer.matrix[i][j] * this.components[j];
      }
      newComponents[i] = sum;
    }

    return new Vector(transformer.rows, (i: number) => newComponents[i]);
  }
  public softmax(): Vector {
    // Compute the exponentials of each component
    const expComponents = this.components.map((value) => Math.exp(value));

    // Sum of the exponentials
    const sumExp = expComponents.reduce((sum, value) => sum + value, 0);

    // Divide each exponential by the sum to get probabilities
    const softmaxed = expComponents.map((value) => value / sumExp);

    return new Vector(this.components.length, (i:number) => softmaxed[i]);
  }
}

export class Matrix {
  public matrix: number[][] = [];
  public rows: number;
  public columns: number;

  public constructor(
    rows: number,
    columns: number,
    genFunc: (row: number, col: number) => number
  ) {
    this.rows = rows;
    this.columns = columns;
    for (let i = 0; i < rows; i++) {
      this.matrix[i] = [];
      for (let j = 0; j < columns; j++) {
        this.matrix[i][j] = genFunc(i, j);
      }
    }
  }

  // Transpose the matrix
  public transpose(): Matrix {
    return new Matrix(this.columns, this.rows, (row, col) => this.matrix[col][row]);
  }

  // Scale the matrix by a scalar
  public scale(scalar: number): Matrix {
    return new Matrix(this.rows, this.columns, (row, col) => this.matrix[row][col] * scalar);
  }

  // Subtract another matrix
  public subtract(other: Matrix): Matrix {
    if (this.rows !== other.rows || this.columns !== other.columns) {
      throw new Error("Matrix dimensions must match for subtraction.");
    }
    return new Matrix(this.rows, this.columns, (row, col) => this.matrix[row][col] - other.matrix[row][col]);
  }
}
