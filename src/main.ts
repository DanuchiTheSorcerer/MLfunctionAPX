import { NeuralNetwork } from './neuralNetwork.ts';
import { Matrix, Vector } from './linearAlgebra.ts';

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

alert(nn.forwards(v1).components[0]);
