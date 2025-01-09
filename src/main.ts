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



function updateScreen() { 
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
  <p>${epoch}</p>
  </div>
`;
  requestAnimationFrame(()=>updateScreen)
}

let nn = new NeuralNetwork([1, 20, 20, 1]); // [input, hidden1, hidden2, output]
let epoch:number = 0
let trainingIn: Vector[] = [];
let trainingOut: Vector[] = [];

updateScreen();


for (let i = 0; i < 1000; i++) {
    let x = Math.random()*10 - 5;
    trainingIn.push(new Vector(1, () => x));
    trainingOut.push(new Vector(1, () => x^2)); // Replace with cos(x) or a polynomial for other functions
}

// Train the network
for (epoch = 0; epoch < 1000; epoch++) {
    nn.train(trainingIn, trainingOut, 0.1); // Adjust learning rate as needed
}

// Test the network
let testInput = new Vector(1, () => 0.5); // Example input
alert("Predicted: "+ nn.forwards(testInput).components[0]);
alert("Actual: "+ 0.25);