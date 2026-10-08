import fs from 'node:fs/promises';
import path from 'node:path';
export const ink='#26344B', blue='#3569A9', teal='#237F79', orange='#B26A23';
const node=(id,label,x,y,w,h,color=blue,fill='#EDF3FA',size=18)=>({id,label,x,y,w,h,color,fill,size,kind:'rect'});
const label=(id,label,x,y,w,h,size=18,color=ink,bold=false)=>({id,label,x,y,w,h,size,color,bold,kind:'text'});
export const pipeline={width:850,height:342,nodes:[
 label('a','(a) Equivalent reparameterization',0,0,276,53,19,ink,true),
 label('b','(b) Weight VQ',310,0,210,28,19,ink,true),
 label('c','(c) Action inference',633,0,215,28,19,ink,true),
 node('x','X',0,65,54,54),
 node('xs','Scale + Hadamard\nX̃ = XD⁻¹H',87,59,184,66),
 node('aq','BF16 first\nA8 / A4 later',319,59,138,66,blue,'#EDF3FA',17),
 node('w','W',0,191,54,54,orange,'#FCF2E6'),
 node('ws','Paired transform\nW̃ = WDH',87,185,184,66,orange,'#FCF2E6'),
 node('vq','Sensitive codebook\nC + vector indices',319,179,188,78,teal,'#E9F4F0',18),
 node('decode','Reconstruct\nŴ',542,186,112,65,teal,'#E9F4F0',18),
 node('linear','Linear',694,117,79,60,ink,'#F5F6F8',18),
 node('action','Action',789,119,61,57,ink,'#FFFFFF',16),
 node('metric','Action sensitivity\nM from calibration',319,287,188,55,teal,'#FFFFFF',17),
 label('acttag','Aim: reduce activation outliers',87,137,240,28,16,blue),
 label('wtag','Scale shift to the weight branch',87,263,237,28,16,orange),
 label('cal','Dashed path: offline calibration',556,293,294,36,15,teal)
],edges:[
 {from:'x',to:'xs'}, {from:'xs',to:'aq'},
 {from:'aq',to:'linear',fromSide:'right',toSide:'top',kind:'elbow'},
 {from:'w',to:'ws',color:orange}, {from:'ws',to:'vq',color:orange},
 {from:'vq',to:'decode',color:teal},
 {from:'decode',to:'linear',fromSide:'right',toSide:'bottom',kind:'elbow',color:teal},
 {from:'linear',to:'action'},
 {from:'metric',to:'vq',fromSide:'top',toSide:'bottom',dashed:true,color:teal}
]};
export const geometry={width:850,height:348,nodes:[
 label('ga','(a) A two-dimensional weight block',0,0,434,30,19,ink,true),
 label('gb','(b) Action-weighted distortion',482,0,368,30,19,ink,true),
 {id:'region',label:'',x:110,y:77,w:82,h:246,color:'#86BFB2',fill:'#EFF8F5',kind:'ellipse',size:18},
 {id:'w0',label:'',x:145,y:194,w:13,h:13,color:ink,fill:ink,kind:'ellipse',size:18},
 {id:'cL2',label:'',x:204,y:194,w:13,h:13,color:orange,fill:orange,kind:'ellipse',size:18},
 {id:'cM',label:'',x:145,y:76,w:13,h:13,color:teal,fill:teal,kind:'ellipse',size:18},
 label('wlabel','w = (0, 0)',18,230,153,27,17,ink),
 label('l2label','c₁ = (0.3, 0)\nEuclidean nearest',233,240,220,59,17,orange),
 label('mlabel','c₂ = (0, 0.6)\nAction-weighted nearest',225,70,239,63,17,teal),
 label('xaxis','Sensitive direction',244,163,209,28,16,ink),
 label('yaxis','Less sensitive',0,37,194,28,16,ink),
 label('formula','dₘ(v,c) = (v−c)ᵀM(v−c)',482,63,368,41,25,ink,true),
 label('metricnote','M = diag(9, 1) in this illustration',482,118,368,35,19,ink),
 label('l2dist','Euclidean distortion\nc₁: 0.09       c₂: 0.36',482,176,368,68,22,orange),
 label('mdist','Action-weighted distortion\nc₁: 0.81       c₂: 0.36',482,268,368,68,22,teal),
 label('ellipsenote','Ellipse: equal action-weighted error',0,323,457,25,15,teal),
 {id:'xstart',label:'',x:151,y:200,w:1,h:1,color:ink,fill:'none',kind:'text'},
 {id:'xend',label:'',x:444,y:200,w:1,h:1,color:ink,fill:'none',kind:'text'},
 {id:'ystart',label:'',x:151,y:200,w:1,h:1,color:ink,fill:'none',kind:'text'},
 {id:'yend',label:'',x:151,y:36,w:1,h:1,color:ink,fill:'none',kind:'text'}
],edges:[
 {from:'xstart',to:'xend',color:'#8895A4',width:1},
 {from:'ystart',to:'yend',color:'#8895A4',width:1},
 {from:'w0',to:'cL2',color:orange,width:2},
 {from:'w0',to:'cM',fromSide:'top',toSide:'bottom',color:teal,width:2}
]};
export const baseline={width:850,height:334,nodes:[
 label('stage1','Stage 1  ·  Imagine future video',0,0,405,29,20,ink,true),
 label('stage2','Stage 2  ·  Infer actions from frozen video',437,0,413,29,20,ink,true),
 node('obs','Observation\n+ instruction',0,67,124,69,blue,'#EDF3FA',18),
 node('video','VideoDiT\nvideo denoising',165,60,185,83,blue,'#EDF3FA',21),
 node('cache','Frozen video\nK / V context',437,60,172,83,blue,'#EDF3FA',21),
 node('actiondit','ActionDiT\naction denoising',437,204,172,83,teal,'#E9F4F0',21),
 label('scope','First VQ target:\nindependent\nActionDiT',657,60,193,104,21,teal,true),
 label('frozen','Hold the video branch in BF16 for the first pilot',0,160,379,50,18,blue),
 node('proprio','Proprioception\n+ action noise',165,211,185,69,ink,'#F5F6F8',18),
 node('out','Action chunk',657,213,193,65,ink,'#F5F6F8',21),
 label('extension','Transfer after the Fast-WAM pilot:   LingBot-VA  →  Cosmos-Policy',0,303,850,30,21,ink,true)
],edges:[
 {from:'obs',to:'video',color:blue},
 {from:'video',to:'cache',color:blue},
 {from:'cache',to:'actiondit',fromSide:'bottom',toSide:'top',color:teal},
 {from:'proprio',to:'actiondit',color:ink},
 {from:'actiondit',to:'out',color:teal}
]};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll('\n','&#xa;');
export async function writeDrawio(dir){
 await fs.mkdir(dir,{recursive:true});
 for(const [name,fig] of [['fast-wam-idm-baseline',baseline],['weight-vq-pipeline',pipeline],['action-sensitive-codebook',geometry]]){
  const cells=fig.nodes.map(n=>{
   const style=`${n.kind==='text'?'text;strokeColor=none;fillColor=none;':`${n.kind==='ellipse'?'ellipse;':'rounded=0;'}strokeColor=${n.color};fillColor=${n.fill};strokeWidth=1.4;`}html=1;whiteSpace=wrap;fontFamily=Arial;fontSize=${n.size??18};fontColor=${n.color};fontStyle=${n.bold?1:0};align=${n.kind==='text'?'left':'center'};verticalAlign=middle;spacing=5;`;
   return `<mxCell id="${n.id}" value="${esc(n.label)}" style="${style}" vertex="1" parent="1"><mxGeometry x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" as="geometry"/></mxCell>`;
  }).concat(fig.edges.map((e,i)=>`<mxCell id="e${i}" value="" style="${e.kind==='elbow'?'edgeStyle=orthogonalEdgeStyle;':''}rounded=0;html=1;strokeColor=${e.color??ink};strokeWidth=${e.width??1.7};endArrow=classic;endSize=6;${e.dashed?'dashed=1;':''}${e.fromSide?`exitX=${e.fromSide==='right'?1:e.fromSide==='left'?0:.5};exitY=${e.fromSide==='top'?0:e.fromSide==='bottom'?1:.5};`:''}${e.toSide?`entryX=${e.toSide==='right'?1:e.toSide==='left'?0:.5};entryY=${e.toSide==='top'?0:e.toSide==='bottom'?1:.5};`:''}" edge="1" source="${e.from}" target="${e.to}" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>`));
  await fs.writeFile(path.join(dir,`${name}.drawio`),`<mxGraphModel adaptiveColors="auto" grid="0" page="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>${cells.join('')}</root></mxGraphModel>`,'utf8');
 }
}
await writeDrawio('D:/Downloads/Final Year Project/.codex-tmp/meeting-20261007/method-figures');
