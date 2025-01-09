var j=Object.defineProperty;var q=(s,t,e)=>t in s?j(s,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):s[t]=e;var m=(s,t,e)=>q(s,typeof t!="symbol"?t+"":t,e);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const o of document.querySelectorAll('link[rel="modulepreload"]'))n(o);new MutationObserver(o=>{for(const i of o)if(i.type==="childList")for(const l of i.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&n(l)}).observe(document,{childList:!0,subtree:!0});function e(o){const i={};return o.integrity&&(i.integrity=o.integrity),o.referrerPolicy&&(i.referrerPolicy=o.referrerPolicy),o.crossOrigin==="use-credentials"?i.credentials="include":o.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function n(o){if(o.ep)return;o.ep=!0;const i=e(o);fetch(o.href,i)}})();class h{constructor(t,e){m(this,"components",[]);this.components=[];for(let n=0;n<t;n++)this.components[n]=e(n)}add(t){let e=[];for(let n=0;n<this.components.length;n++)e[n]=t.components[n]+this.components[n];return new h(this.components.length,n=>e[n])}mult(t){let e=[];for(let n=0;n<this.components.length;n++)e[n]=t.components[n]*this.components[n];return new h(this.components.length,n=>e[n])}scale(t){let e=[];for(let n=0;n<this.components.length;n++)e[n]=this.components[n]*t;return new h(this.components.length,n=>e[n])}subtract(t){let e=[];for(let n=0;n<this.components.length;n++)e[n]=this.components[n]-t.components[n];return new h(this.components.length,n=>e[n])}transform(t){if(t.columns!==this.components.length)throw new Error("Matrix columns must match the number of vector components.");let e=[];for(let n=0;n<t.rows;n++){let o=0;for(let i=0;i<t.columns;i++)o+=t.matrix[n][i]*this.components[i];e[n]=o}return new h(t.rows,n=>e[n])}softmax(){const t=this.components.map(o=>Math.exp(o)),e=t.reduce((o,i)=>o+i,0),n=t.map(o=>o/e);return new h(this.components.length,o=>n[o])}}class u{constructor(t,e,n){m(this,"matrix",[]);m(this,"rows");m(this,"columns");this.rows=t,this.columns=e;for(let o=0;o<t;o++){this.matrix[o]=[];for(let i=0;i<e;i++)this.matrix[o][i]=n(o,i)}}transpose(){return new u(this.columns,this.rows,(t,e)=>this.matrix[e][t])}scale(t){return new u(this.rows,this.columns,(e,n)=>this.matrix[e][n]*t)}subtract(t){if(this.rows!==t.rows||this.columns!==t.columns)throw new Error("Matrix dimensions must match for subtraction.");return new u(this.rows,this.columns,(e,n)=>this.matrix[e][n]-t.matrix[e][n])}add(t){if(this.rows!==t.rows||this.columns!==t.columns)throw new Error("Matrix dimensions must match for add.");return new u(this.rows,this.columns,(e,n)=>this.matrix[e][n]+t.matrix[e][n])}}class v{constructor(t){m(this,"weights");m(this,"biases");m(this,"layerSizes");this.layerSizes=t,this.weights=[],this.biases=[];for(let e=0;e<t.length-1;e++){const n=t[e];this.weights.push(new u(t[e+1],t[e],()=>Math.random()*Math.sqrt(2/n)*(Math.random()>.5?1:-1))),this.biases.push(new h(t[e+1],()=>0))}}forwards(t){let e=t;for(let n=0;n<this.weights.length;n++)e=e.transform(this.weights[n]).add(this.biases[n]),e=this.sigmoid(e);return e}sigmoid(t){return new h(t.components.length,e=>Math.tanh(t.components[e]))}sigmoidPrime(t){let e=this.sigmoid(t);return new h(t.components.length,()=>1).subtract(e.mult(e))}train(t,e,n){const o=t.length,i=this.weights.map(r=>new u(r.rows,r.columns,()=>0)),l=this.biases.map(r=>new h(r.components.length,()=>0));for(let r=0;r<o;r++){const O=t[r],z=e[r],p=[],w=[];p[0]=O;for(let c=1;c<this.layerSizes.length;c++){const g=p[c-1].transform(this.weights[c-1]).add(this.biases[c-1]);w[c]=g,p[c]=this.sigmoid(g)}let f=p[this.layerSizes.length-1].subtract(z).mult(this.sigmoidPrime(w[this.layerSizes.length-1]));for(let c=this.layerSizes.length-2;c>=0;c--){const g=new u(this.layerSizes[c+1],this.layerSizes[c],(L,N)=>f.components[L]*p[c].components[N]);i[c]=i[c].add(g),l[c]=l[c].add(f),c>0&&(f=f.transform(this.weights[c].transpose()).mult(this.sigmoidPrime(w[c])))}}for(let r=0;r<this.weights.length;r++)i[r]=i[r].scale(1/o),l[r]=l[r].scale(1/o),this.weights[r]=this.weights[r].subtract(i[r].scale(n)),this.biases[r]=this.biases[r].subtract(l[r].scale(n))}}let y=new v([1,40,40,40,1]),b=[],x=[],P,M=0;function S(s){b=[],x=[];for(let t=0;t<250;t++){let e=Math.random()*2-1;b.push(new h(1,()=>e)),x.push(new h(1,()=>s(e)))}}const C=document.createElement("div");document.body.appendChild(C);const I=[{name:"y = x",func:s=>s},{name:"y = x^2",func:s=>s*s},{name:"y = x^3",func:s=>s*s*s},{name:"y = e^x - 1",func:s=>Math.exp(s)-1},{name:"y = x^2 - x",func:s=>s*s-s},{name:"y = sin(2pi x)",func:s=>Math.sin(2*Math.PI*s)/2},{name:"y = |x|-0.5",func:s=>Math.abs(s)-.5},{name:"y = ln(x-1.1)",func:s=>Math.log(s+1.01)},{name:"y = |tanh(x-0.2)|",func:s=>Math.abs(Math.tanh(s-.2))},{name:"y = 3x^4 - 2x^2",func:s=>3*s*s*s*s-2*s*s},{name:"y = -1 + sqrt(x+1)",func:s=>-1+Math.sqrt(s+1)}];I.forEach(({name:s,func:t})=>{const e=document.createElement("button");e.textContent=s,e.onclick=()=>{y=new v([1,20,20,1]),S(t),console.log(`Selected function: ${s}`),P=t,M=0},C.appendChild(e)});C.appendChild(document.createElement("p"));const d=document.createElement("canvas");d.width=600;d.height=600;document.body.appendChild(d);const a=d.getContext("2d");a.translate(d.width/2,d.height/2);a.scale(d.width/2,-d.height/2);function A(){a.strokeStyle="gray",a.lineWidth=.005,a.beginPath(),a.moveTo(-1,0),a.lineTo(1,0),a.stroke(),a.beginPath(),a.moveTo(0,-1),a.lineTo(0,1),a.stroke()}function E(s,t){a.fillStyle=t,s.forEach(({x:e,y:n})=>{a.beginPath(),a.arc(e,n,.01,0,2*Math.PI),a.fill()})}function T(){M+=1,y.train(b,x,.2);let s=[],t=[];for(let e=-1;e<=1;e+=.01){let n=P(e),o=y.forwards(new h(1,()=>e)).components[0];s.push({x:e,y:n}),t.push({x:e,y:o})}a.clearRect(-1,-1,2,2),A(),E(s,"green"),E(t,"red"),document.getElementsByTagName("p")[0].innerHTML="epoch: "+M,requestAnimationFrame(T)}P=s=>s;S(s=>s);T();