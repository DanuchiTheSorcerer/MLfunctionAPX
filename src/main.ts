import { NeuralNetwork } from './neuralNetwork.ts';
import { Vector } from './linearAlgebra.ts';



// let nn = new NeuralNetwork([10,10,1])

// let v1 = new Vector(10,(i:number)=>{return i})
// let target = new Vector(1,()=>0.5)

// alert(nn.forwards(v1).components[0])
// for (let i = 0;i<1000;i++) {
//   nn.train(v1,target,0.1)
// }
// alert(nn.forwards(v1).components[0])

// Network will have layers 12 20 10 3
//in the input and output layers a [0 or 1, 0 or 1, 0 or 1] vector will be for [R,P,S]
// 1 Engine takes in last few moves and guesses probabilites, submitting expected value determined (randomly chooses rps for first few rounds)
// 2 Player makes a move
// 3 Score is updated
// 4 Engine trained with whatever would have beaten the player as input
// Run another round



// Neural network setup
let nn = new NeuralNetwork([1, 20,20, 1]); // Define layers [input, hidden, output] with 'tanh' activation
let trainingIn: Vector[] = [];
let trainingOut: Vector[] = [];

// Generate training data (evenly distributed between -1 and 1)
function generateTrainingData(func:Function):void {
  trainingIn = [];
  trainingOut = [];
  for (let i = 0; i < 150; i++) {
    let x: number = Math.random() * 2 - 1; // Random x in [-1, 1]
    trainingIn.push(new Vector(1, () => x));
    trainingOut.push(new Vector(1, () => func()));
  }
}


// Canvas setup
const canvas = document.createElement('canvas');
canvas.width = 600;
canvas.height = 600;
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d')!;
ctx.translate(canvas.width / 2, canvas.height / 2); // Center canvas
ctx.scale(canvas.width / 2, -canvas.height / 2); // Scale for Cartesian graph

// Function to draw axes
function drawAxes() {
  ctx.strokeStyle = 'gray';
  ctx.lineWidth = 0.005;

  // X-axis
  ctx.beginPath();
  ctx.moveTo(-1, 0);
  ctx.lineTo(1, 0);
  ctx.stroke();

  // Y-axis
  ctx.beginPath();
  ctx.moveTo(0, -1);
  ctx.lineTo(0, 1);
  ctx.stroke();
}

// Function to draw points
function drawPoints(points: { x: number; y: number }[], color: string) {
  ctx.fillStyle = color;
  points.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(x, y, 0.01, 0, 2 * Math.PI);
    ctx.fill();
  });
}

// Training and visualization
function update() {
  // Train the network for one step
  nn.train(trainingIn, trainingOut, 0.1);

  // Sample points for visualization
  let truePoints: { x: number; y: number }[] = [];
  let predictedPoints: { x: number; y: number }[] = [];
  for (let x = -1; x <= 1; x += 0.01) {
    let yTrue = approximatedFunction(x);
    let yPred = nn.forwards(new Vector(1, () => x)).components[0];
    truePoints.push({ x, y: yTrue });
    predictedPoints.push({ x, y: yPred });
  }

  // Clear canvas
  ctx.clearRect(-1, -1, 2, 2);

  // Draw axes
  drawAxes();

  // Draw points
  drawPoints(truePoints, 'green'); // True function (x^2)
  drawPoints(predictedPoints, 'red'); // Predicted points

  // Continue animation
  requestAnimationFrame(update);
}

function approximatedFunction(x:number):number {
  return x
  //functions to be added
  //x
  //x^2
  //x^3
  //e^x -1
  //x*x -x
  // one that you get to choose
}

// Start animation
update();









// Create buttons for function selection
const buttonContainer = document.createElement('div');
document.body.appendChild(buttonContainer);

// List of functions to approximate
const functions: { name: string; func: (x: number) => number }[] = [
  { name: "y = x", func: (x) => x },
  { name: "y = x^2", func: (x) => x * x },
  { name: "y = x^3", func: (x) => x * x * x },
  { name: "y = e^x - 1", func: (x) => Math.exp(x) - 1 },
  { name: "y = x^2 - x", func: (x) => x * x - x },
];

// Create buttons for each function
functions.forEach(({ name, func }) => {
  const button = document.createElement('button');
  button.textContent = name;
  button.onclick = () => {
    // Reset the neural network
    nn = new NeuralNetwork([1, 20, 20, 1]);

    // Generate new training data
    generateTrainingData(func);
    approximatedFunction = func
    // Log the selected function
    console.log(`Selected function: ${name}`);
  };
  buttonContainer.appendChild(button);
});