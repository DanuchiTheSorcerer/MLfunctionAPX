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
}
