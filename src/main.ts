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



let nn = new NeuralNetwork([1, 40,40,40, 1]);
let trainingIn: Vector[] = [];
let trainingOut: Vector[] = [];
let currentFunction:Function
let epoch:number = 0

function generateTrainingData(func:Function):void {
  trainingIn = [];
  trainingOut = [];
  for (let i = 0; i < 250; i++) {
    let x: number = Math.random() * 2 - 1; 
    trainingIn.push(new Vector(1, () => x));
    trainingOut.push(new Vector(1, () => func(x)));
  }
}

const buttonContainer = document.createElement('div');
document.body.appendChild(buttonContainer);

const functions: { name: string; func: (x: number) => number }[] = [
  { name: "y = x", func: (x) => x },
  { name: "y = x^2", func: (x) => x * x },
  { name: "y = x^3", func: (x) => x * x * x },
  { name: "y = e^x - 1", func: (x) => Math.exp(x) - 1 },
  { name: "y = x^2 - x", func: (x) => x * x - x },
  { name: "y = sin(2pi x)", func: (x) => Math.sin(2*Math.PI*x)/2 },
  { name: "y = |x|-0.5", func: (x) => Math.abs(x)-0.5 },
  { name: "y = ln(x-1.1)", func: (x) => Math.log(x+1.01) },
  { name: "y = |tanh(x-0.2)|", func: (x) => Math.abs(Math.tanh(x-0.2)) },
  { name: "y = 3x^4 - 2x^2", func: (x) => 3*x*x*x*x - 2*x*x},
  { name: "y = -1 + sqrt(x+1)", func: (x) => -1 + Math.sqrt(x+1)}
];

functions.forEach(({ name, func }) => {
  const button = document.createElement('button');
  button.textContent = name;
  button.onclick = () => {
    nn = new NeuralNetwork([1, 20, 20, 1]);

    generateTrainingData(func);
    console.log(`Selected function: ${name}`);
    currentFunction = func
    epoch = 0
  };
  buttonContainer.appendChild(button);
});
buttonContainer.appendChild(document.createElement('p'))


const canvas = document.createElement('canvas');
canvas.width = 600;
canvas.height = 600;
document.body.appendChild(canvas);
const ctx = canvas.getContext('2d')!;
ctx.translate(canvas.width / 2, canvas.height / 2);
ctx.scale(canvas.width / 2, -canvas.height / 2); 

function drawAxes() {
  ctx.strokeStyle = 'gray';
  ctx.lineWidth = 0.005;

  ctx.beginPath();
  ctx.moveTo(-1, 0);
  ctx.lineTo(1, 0);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(0, -1);
  ctx.lineTo(0, 1);
  ctx.stroke();
}

function drawPoints(points: { x: number; y: number }[], color: string) {
  ctx.fillStyle = color;
  points.forEach(({ x, y }) => {
    ctx.beginPath();
    ctx.arc(x, y, 0.01, 0, 2 * Math.PI);
    ctx.fill();
  });
}



function update() {
  epoch += 1

  nn.train(trainingIn, trainingOut, 0.2);

  

  let truePoints: { x: number; y: number }[] = [];
  let predictedPoints: { x: number; y: number }[] = [];
  for (let x = -1; x <= 1; x += 0.01) {
    let yTrue = currentFunction(x);
    let yPred = nn.forwards(new Vector(1, () => x)).components[0];
    truePoints.push({ x, y: yTrue });
    predictedPoints.push({ x, y: yPred });
  }

  ctx.clearRect(-1, -1, 2, 2);

  drawAxes();

  drawPoints(truePoints, 'green');
  drawPoints(predictedPoints, 'red');
  document.getElementsByTagName('p')[0].innerHTML = "epoch: " + epoch
  requestAnimationFrame(update);
}

currentFunction = (x:number) => {return x}

generateTrainingData((x:number)=>{return x})


update();

