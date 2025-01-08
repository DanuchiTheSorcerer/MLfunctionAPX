import { NeuralNetwork } from './neuralNetwork.ts';
import { Vector } from './linearAlgebra.ts';

function updateScreen() {
  document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div>
  hi
  </div>
`;
}
updateScreen();

let nn = new NeuralNetwork([2, 3,4, 1]);
let v1 = new Vector(2, (i: number) => {
  return i;
});
let target = new Vector(1,()=>0)

alert(nn.forwards(v1).components[0]);
for (let i = 0;i<100000;i++) {
  nn.train(v1,target,0.1)
}
alert(nn.forwards(v1).components[0]);

// Network will have layers 12 20 10 3
//in the input and output layers a [0 or 1, 0 or 1, 0 or 1] vector will be for [R,P,S]
// 1 Engine takes in last few moves and guesses probabilites, submitting expected value determined (randomly chooses rps for first few rounds)
// 2 Player makes a move
// 3 Score is updated
// 4 Engine trained with whatever would have beaten the player as input
// Run another round

let engine = new NeuralNetwork([12,20,10,3])
let round:number = 0
let playerMoves: number[][] = []
let engineMoves: number[][] = []

function runRound(playerDecision:string): void {
  let engineMove: number[] = []
  let playerMove: number[] = []
  //turn player decision into an array
  if (playerDecision === "Rock") {
    playerMove = [1,0,0];
  } else if (playerDecision === "Paper") {
    playerMove = [0,1,0];
  } else if (playerDecision === "Scissors") {
    playerMove = [0,0,1];

  }

  //get the data to forwards and backwards propagate by turning it into a vector
  let trainingData: number[] = []

  for (let i = playerMoves.length-2;i<playerMoves.length;i++) {
    for (let j = 0;j<playerMoves[i].length;j++) {
      trainingData.push(playerMoves[i][j])
      trainingData.push(engineMoves[i][j])
    }
  }
  let trainingInput = new Vector(12,(i:number) => {return trainingData[i]})
  //engine decides its move
  if (round < 5) {
    let randChoice:number = Math.floor(Math.random() * 3)
    if (randChoice==0) {
      engineMove = [1,0,0]
    } else if (randChoice==1) {
      engineMove = [0,1,0]
    } else {
      engineMove = [0,0,1]
    }
  } else {
    engine.forwards(trainingInput)
  }

  //scores are updated
  

  //engine is trained and moves are push for the next round
  if (playerMove[0]) {
    target = new Vector(3,(i:number) => {return [0,1,0][i]})
  } else if (playerMove[1]){
    target = new Vector(3,(i:number) => {return [0,0,1][i]})
  } else if (playerMove[2]) {
    target = new Vector(3,(i:number) => {return [1,0,0][i]})
  }
  engine.train(trainingInput,target,0.1)
  playerMoves.push(playerMove)
  engineMoves.push(engineMove)
}