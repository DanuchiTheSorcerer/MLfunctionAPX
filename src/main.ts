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

let nn = new NeuralNetwork([2, 3, 1]);
let v1 = new Vector(2, (i: number) => {
  return i;
});
let target = new Vector(1,()=>1)

alert(nn.forwards(v1).components[0]);
for (let i = 0;i<1000000;i++) {
  nn.train(v1,target,0.1)
}
alert(nn.forwards(v1).components[0]);
