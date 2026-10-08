import fs from 'node:fs/promises';
import path from 'node:path';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
import {canvas} from '../meeting-20261007-revision/figures.mjs';
const ROOT='D:/Downloads/Final Year Project',TMP=path.join(ROOT,'.codex-tmp/meeting-20261007-strict');
const SOURCE=path.join(ROOT,'Weekly Slides/2026-10-07_WAM_Quantization_Progress_Revised.pptx');
const ASSETS=path.join(ROOT,'Weekly Slides/2026-10-07_Figure_Designs');
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
await fs.mkdir(TMP,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
if(p.slides.items.length!==12)throw new Error('The user-edited source must have 12 slides.');
if(process.argv.includes('--inspect')){for(const n of [10,11]){const png=await p.export({slide:p.slides.items[n-1],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,`source-${n}.png`),Buffer.from(await png.arrayBuffer()));}console.log('Rendered current source slides 10 and 11');process.exit(0);}
const K=936/1862,F={left:12,top:76,width:936,height:845*K};
const C={navy:'#0B2255',ink:'#16283F',gray:'#59687A',blue:'#7DA6D3',blueFill:'#EDF4FD',gold:'#A7600C',goldFill:'#FCF4E5',teal:'#006C78',tealFill:'#EDF7F8',red:'#C92048',redFill:'#FDEFF2'};
function kit(s){
 const g=canvas(s,F.left,F.top),{S,T,L,A}=g;
 function box(x,y,w,h,fill='none',stroke='none',lw=1,round=false,dash=false){const q=S(x*K,y*K,w*K,h*K,fill,stroke,lw*K,round?'roundRect':'rect',dash);q.name='Editable reference element';return q;}
 function text(t,x,y,w,h,{size=23,bold=false,italic=false,color=C.navy,align='left',middle=false}={}){const q=T(t,x*K,y*K,w*K,h*K,{size:size*K,bold,italic,color,align,middle});q.text.wrap='none';q.name='Editable reference text: '+t.slice(0,50);return q;}
 function line(x1,y1,x2,y2,{color=C.navy,width=1.5,arrow=false,dash=false}={}){const dx=x2-x1,dy=y2-y1;
  const q=S(Math.min(x1,x2)*K,Math.min(y1,y2)*K,Math.max(Math.abs(dx)*K,.01),Math.max(Math.abs(dy)*K,.01),'none',color,width*K,'line',dash);q.position={...q.position,horizontalFlip:dx<0,verticalFlip:dy<0};q.name='Reference line';
  if(arrow){const len=Math.hypot(dx,dy),ah=Math.max(7,width*4),aw=Math.max(6,width*3.3),ux=dx/len,uy=dy/len;const a=S((x2-ux*ah/2-aw/2)*K,(y2-uy*ah/2-ah/2)*K,aw*K,ah*K,color,'none',0,'triangle');a.position={...a.position,rotation:Math.atan2(dy,dx)*180/Math.PI+90};a.name='Reference arrowhead';}return q;}
 function dot(x,y,r=9,color=C.navy){const q=g.D(x*K,y*K,color,r*K);q.name='Reference point';return q;}
 function ellipse(x,y,rx,ry,{color=C.teal,fill='none',width=1,dash=false,rotation=0}={}){const q=g.E(x*K,y*K,rx*K,ry*K,{stroke:color,fill,width:width*K,dash});if(rotation)q.position={...q.position,rotation};return q;}
 function matrix(x,y,w,h,r,c,palette='blue',key){const cells=colors[key];for(let i=0;i<r;i++)for(let j=0;j<c;j++)box(x+j*w/c,y+i*h/r,w/c,h/r,cells?.[i*c+j]??(palette==='gold'?'#EBC07C':'#9BBBDC'),'#FFFFFF',.9);box(x,y,w,h,'none',palette==='gold'?'#AC7526':'#6885A5',1);}
 function ellipsis(x,y,w=42,h=25){text('⋯',x,y,w,h,{size:28});}
 function dim(x1,y1,x2,y2,label,lx,ly,w=90,{vertical=false}={}){line(x1,y1,x2,y2,{arrow:true,width:1.5});line(x2,y2,x1,y1,{arrow:true,width:1.5});const q=text(label,lx,ly,w,27,{size:21,italic:true,align:'center'});if(vertical)q.position={...q.position,rotation:270};}
 return{box,text,line,dot,ellipse,matrix,ellipsis,dim};
}
const colors=JSON.parse(await fs.readFile(path.join(TMP,'matrix-colors.json'),'utf8'));
function method(s){const{box,text,line,dot,ellipse,matrix,ellipsis,dim}=kit(s);
 line(700,20,700,822,{color:'#A5B5C6',dash:true,width:1});line(1412,20,1412,822,{color:'#A5B5C6',dash:true,width:1});
 text('(a) Equivalent conditioning',24,17,660,44,{size:34,bold:true});text('Reparameterize activations and weights (no change before quantization)',24,61,660,31,{size:23});
 text('(b) Action-sensitive weight VQ',725,17,650,44,{size:34,bold:true});text('Quantize transformed weights with an action-aware metric',725,61,670,31,{size:23});
 text('(c) Action inference',1435,17,410,44,{size:34,bold:true});text('Quantized weights for low-bit inference',1435,61,410,31,{size:23});
 text('Activations  X ∈ ℝᵀˣᵈ',32,123,337,32,{size:24,bold:true});text('outlier\nchannels',158,153,102,43,{size:20,bold:true,color:C.gold,align:'center'});
 line(155,208,155,214,{color:C.gold});line(155,204,217,204,{color:C.gold,dash:true});line(217,204,217,214,{color:C.gold});
 matrix(74,216,143,114,6,7,'blue','m-act');dim(60,215,60,334,'T (tokens)',3,256,111,{vertical:true});dim(74,344,217,344,'d (channels)',95,351,151);ellipsis(225,254);
 box(259,226,174,89,C.blueFill,C.blue,1.4);text('Scale + Hadamard',274,241,150,31,{size:23,align:'center'});text('X′ = XD⁻¹H',279,274,149,34,{size:26,italic:true,align:'center'});line(434,272,462,272,{color:C.blue,arrow:true,width:4});
 text('Conditioned activations',430,123,259,32,{size:24,bold:true});text('X′ ∈ ℝᵀˣᵈ',430,153,244,32,{size:25,italic:true});matrix(504,216,130,114,6,7,'blue','m-cond');dim(489,215,489,334,'T',463,276,20);dim(504,344,634,344,'d',535,351,57);ellipsis(642,254);
 text('Weights  W ∈ ℝⁿˣᵈ',32,418,370,32,{size:25,bold:true});matrix(74,484,145,114,6,8,'gold','m-w');dim(60,484,60,601,'n (output dim)',-7,530,133,{vertical:true});dim(74,611,219,611,'d (input dim)',83,619,171);line(231,540,256,540,{color:'#E7A07D',arrow:true,width:4});
 box(259,494,174,90,'#FFF7EF','#E7A07D',1.5);text('Paired transform',271,507,153,30,{size:23,color:'#9C420D',align:'center'});text('W′ = WDH',280,541,147,33,{size:26,color:'#9C420D',italic:true,align:'center'});line(434,540,462,540,{color:'#E7A07D',arrow:true,width:4});
 text('Transformed weights',461,418,227,32,{size:24,bold:true});text('W′ ∈ ℝⁿˣᵈ',461,447,207,34,{size:25,italic:true});matrix(502,489,139,105,5,8,'gold','m-wprime');dim(488,489,488,597,'n',461,535,23);dim(502,609,641,609,'d',545,616,64);ellipsis(648,521);
 box(24,685,660,119,'#F0F4F8','none',0,true);text('X′W′ᵀ = XWᵀ   when  HHᵀ = I  and D is invertible.',60,708,602,38,{size:27,italic:true,align:'center'});text('Row-vector convention:   Y = XWᵀ.',82,756,570,35,{size:25,align:'center'});
 text('Row-wise vector quantization (per output row)',730,114,668,35,{size:24,bold:true});text('wᵢ′ ∈ ℝ¹ˣᵈ',772,153,246,31,{size:24,italic:true});matrix(770,188,98,23,1,5,'gold','m-row1');matrix(872,188,98,23,1,5,'blue','m-row2');matrix(1034,188,98,23,1,5,'blue','m-row3');ellipsis(988,179);
 for(const[x,w,t]of[[770,98,'block 1'],[872,98,'block 2'],[1034,98,'block K']]){line(x,219,x+6,228,{width:1});line(x+6,228,x+w-6,228,{width:1});line(x+w-6,228,x+w,219,{width:1});text(t,x,237,w,29,{size:21,align:'center'});}
 text('Split each row into K small vectors',788,264,372,31,{size:22,align:'center'});line(953,294,953,319,{color:'#165689',arrow:true,width:4});
 box(714,326,469,233,C.tealFill,'#3E9EAC',1.2,true,true);text('Sensitive codebook C + vector indices z',731,337,437,31,{size:24,bold:true,color:C.teal});
 text('w ∈ ℝ¹ˣᵐ',733,417,124,30,{size:24,italic:true});matrix(733,448,90,22,1,5,'gold','m-vector');line(849,458,928,458,{color:'#1876A4',arrow:true,width:2});text('arg min',853,425,74,22,{size:20,align:'center'});text('z',878,445,22,17,{size:14,italic:true});text('(w−c_z)ᵀM(w−c_z)',799,479,163,28,{size:20,italic:true});
 text('C ∈ ℝⱽˣᵐ',961,376,147,27,{size:24,italic:true});matrix(960,403,100,54,2,5,'blue','m-code1');ellipsis(995,458,26,30);matrix(960,493,100,22,1,5,'blue','m-code2');text('codebook (V × m)',932,525,167,26,{size:21,align:'center'});text('index  z',1097,375,90,25,{size:21,italic:true});
 for(const[y,v]of[[403,'3'],[432,'1'],[493,'7']]){box(1112,y,38,25,'#F8FCFF','#84AEDB',1);text(v,1113,y,36,25,{size:20,align:'center',middle:true});}text('⋮',1115,460,31,28,{size:24,align:'center'});
 box(1195,168,205,286,C.redFill,'#ED3F64',1.2,true);text('Anisotropic metric M',1206,180,186,30,{size:20.5,bold:true,color:C.red,align:'center'});
 ellipse(1282,321,44,82,{color:'#E86378',rotation:44});ellipse(1282,321,31,64,{color:'#ED7384',rotation:44});ellipse(1282,321,16,40,{color:'#F3BBC2',fill:'#F8C6CB',rotation:44});line(1214,341,1356,341,{color:'#344F75',arrow:true});line(1251,374,1251,231,{color:'#344F75',arrow:true});line(1282,321,1318,286,{color:'#603445',arrow:true,width:2});dot(1282,321,4,'#9E5867');text('v₂',1254,210,40,29,{size:24,italic:true});text('v₁',1364,329,30,29,{size:24,italic:true});text('M',1354,239,33,30,{size:28,italic:true,color:C.red});text('Sensitivity-guided\nassignments',1214,400,172,47,{size:22,color:C.red,align:'center'});
 box(741,605,321,74,C.tealFill,C.teal,1.4,true);text('Action sensitivity  M ∈ ℝᵐˣᵐ',759,618,284,29,{size:22,bold:true,color:C.teal,align:'center'});text('(estimated from calibration)',756,648,290,27,{size:22,color:C.teal,align:'center'});line(901,602,901,562,{color:C.teal,arrow:true,dash:true,width:2.2});
 line(1143,565,1143,595,{color:'#13578C',arrow:true,width:4});text('Reconstruct quantized weights',1102,597,302,33,{size:23,bold:true});text('Ŵ′ ∈ ℝⁿˣᵈ',1137,626,221,30,{size:25,italic:true});matrix(1135,656,195,66,3,9,'gold','m-recon');ellipsis(1347,672);
 box(823,759,520,60,'#FFFFFF',C.teal,1.6,true,true);text('Calibration feedback (offline)',946,770,395,26,{size:21,bold:true,color:'#345075'});text('Measure action sensitivity from final actions and update M, C',847,796,482,24,{size:20,color:'#4C6483',align:'center'});line(900,758,900,682,{color:C.teal,arrow:true,dash:true,width:2.3});line(1344,789,1497,789,{color:C.teal,dash:true,width:2.3});
 text('Quantized activations X′',1446,141,209,28,{size:19,bold:true});box(1455,167,170,60,'#E7F3FF','none',0,true);text('BF16 for first pilot\nA8 / A4 later',1464,172,153,52,{size:21,align:'center'});
 text('Quantized weights Ŵ′',1668,141,182,28,{size:19,bold:true});matrix(1471,239,129,89,5,8,'blue','m-inf-x');matrix(1697,239,121,89,5,7,'gold','m-inf-w');dim(1458,239,1458,332,'T',1425,278,28);dim(1471,342,1600,342,'d',1511,348,64);dim(1683,240,1683,332,'n',1653,279,30);dim(1697,342,1818,342,'d',1737,348,48);ellipsis(1608,271);ellipsis(1827,271);
 line(1524,372,1524,441,{width:2.4});line(1524,441,1557,441,{arrow:true,width:2.4});line(1768,372,1768,441,{width:2.4});line(1768,441,1735,441,{arrow:true,width:2.4});box(1561,399,172,91,'#F0F3F8',C.navy,2);text('Linear',1609,417,82,31,{size:26,bold:true,align:'center'});text('Y = X′Ŵ′ᵀ',1590,453,123,31,{size:25,italic:true,align:'center'});line(1647,493,1647,533,{arrow:true,width:2.5});
 text('Action chunk  Y ∈ ℝᵀˣⁿ',1525,540,300,31,{size:24,bold:true});matrix(1575,585,140,89,5,8,'blue','m-action');dim(1560,585,1560,680,'T',1531,621,29);dim(1575,688,1715,688,'n',1628,698,34);ellipsis(1727,618);
 box(1502,760,293,60,'#FFFFFF',C.teal,1.6,true,true);text('Measure final-action sensitivity',1521,773,263,27,{size:20,color:'#345075',align:'center'});text('(e.g., ∂a/∂x, rollout-based)',1524,797,258,24,{size:20,color:'#345075',align:'center'});line(1651,719,1651,759,{color:C.teal,dash:true,width:2.2});
}
function geometry(s){const{box,text,line,dot,ellipse,matrix,dim}=kit(s);
 line(1277,14,1277,814,{color:'#71849A',width:1});
 text('(a) Euclidean assignment (L₂)',24,13,586,44,{size:34,bold:true});text('Nearest in Euclidean distance',80,56,504,33,{size:27,color:C.gray});
 text('(b) Action-weighted assignment',658,13,604,44,{size:34,bold:true});text('Nearest under action-sensitive metric',721,56,550,33,{size:27,color:C.gray});
 text('(c) Metric and codebook geometry',1300,13,550,44,{size:34,bold:true});
 function plot(x,weighted){const ox=x+282,oy=365;line(x+81,578,x+511,578,{arrow:true,color:'#536477',width:1.3});line(x+81,578,x+81,117,{arrow:true,color:'#536477',width:1.3});line(ox,576,ox,133,{arrow:true,color:'#536477',width:1.3});
  for(const[v,px]of[[-.6,ox-173],[-.3,ox-86],[0,ox],[.3,ox+86],[.6,ox+173]]){line(px,578,px,585,{color:'#536477',width:1});text(String(v).replace('-','−'),px-27,594,62,28,{size:22,align:'center',color:C.ink});}
  for(const[v,py]of[[.6,195],[.3,280],[0,365],[-.3,449],[-.6,532]]){line(x+74,py,x+81,py,{color:'#536477',width:1});text(String(v).replace('-','−'),x+10,py-13,59,29,{size:22,align:'right',color:C.ink});}
  text('Less sensitive\n(direction 2)',x+90,108,181,48,{size:22,color:C.ink});text('More sensitive\n(direction 1)',x+465,516,167,49,{size:22,color:C.ink});
  if(weighted){ellipse(ox,oy,76,177,{color:'#3299A6',fill:'#E7F3F5',dash:true,width:1});line(ox,oy-9,ox,oy-165,{color:C.teal,arrow:true,width:2.2});text('(v − w)ᵀM(v − w) = 0.36',x+365,207,276,37,{size:23,color:C.teal,italic:true});line(x+383,239,ox+65,271,{color:'#68ABB3',arrow:true,width:1.4});}
  else{ellipse(ox,oy,130,130,{color:'#D0AA4D',fill:'#FDF9EC',dash:true,width:1});text('‖v − w‖₂ = 0.6',x+399,213,220,36,{size:26,color:C.gold,italic:true});line(x+393,238,ox+86,265,{color:'#C7A754',arrow:true,width:1.3});}
  line(ox+10,oy,ox+119,oy,{color:C.gold,arrow:true,width:2.6});dot(ox,oy,9,C.ink);dot(ox+130,oy,10,'#BE7103');dot(ox,oy-176,9,C.teal);text('w = (0, 0)',x+191,382,169,31,{size:25,italic:true,color:C.ink});text('c₁ = (0.3, 0)',x+429,335,180,34,{size:25,color:C.gold,italic:true});text('c₂ = (0, 0.6)',x+308,156,191,34,{size:25,color:C.teal,italic:true});
 }
 plot(0,false);plot(636,true);
 box(24,644,584,170,'#FCF8F0','none',0,true);text('Euclidean squared distances',45,660,440,32,{size:25,bold:true,color:C.gold});text('d₂(w, c₁) = ‖w − c₁‖₂² = 0.09',53,711,373,37,{size:25,italic:true,color:C.ink});text('d₂(w, c₂) = ‖w − c₂‖₂² = 0.36',53,758,373,37,{size:25,italic:true,color:C.ink});line(432,693,432,798,{color:'#93A0AE',width:1});text('L₂ chooses',463,705,145,29,{size:25,color:C.ink});box(490,742,65,50,'#F8E9CD','none',0,true);text('c₁',497,741,56,45,{size:34,bold:true,italic:true,color:C.gold,align:'center'});
 box(630,644,626,170,'#EFF7F8','none',0,true);text('Action-weighted squared distances (M = diag(9, 1))',651,660,591,35,{size:25,bold:true,color:C.teal});text('dₘ(w, c₁) = (w − c₁)ᵀM(w − c₁) = 0.81',652,712,444,36,{size:24,italic:true,color:C.ink});text('dₘ(w, c₂) = (w − c₂)ᵀM(w − c₂) = 0.36',652,759,444,36,{size:24,italic:true,color:C.ink});line(1093,693,1093,798,{color:'#93A0AE',width:1});text('Action-aware\nVQ chooses',1107,700,136,47,{size:23,bold:true,align:'center',color:C.ink});box(1146,752,65,47,'#CEE9EE','none',0,true);text('c₂',1151,752,56,43,{size:34,bold:true,italic:true,color:C.teal,align:'center'});
 box(1298,72,550,254,'#F5F7FA','none',0,true);text('Action-weighted distortion',1319,89,505,33,{size:26,bold:true,color:C.ink});text('dₘ(v, c) = (v − c)ᵀM(v − c)',1319,125,507,45,{size:32,italic:true,color:C.ink});text('M =',1463,216,66,38,{size:28,italic:true,color:C.ink});text('9     0\n0     1',1541,188,73,75,{size:25,color:C.ink});
 line(1530,190,1530,264,{color:C.ink,width:1.6});line(1530,190,1539,190,{color:C.ink,width:1.6});line(1530,264,1539,264,{color:C.ink,width:1.6});line(1613,190,1613,264,{color:C.ink,width:1.6});line(1605,190,1613,190,{color:C.ink,width:1.6});line(1605,264,1613,264,{color:C.ink,width:1.6});
 text('higher penalty\n(more sensitive)',1674,180,170,48,{size:23,color:C.gold});line(1665,201,1625,203,{color:C.gold,arrow:true,width:2.4});text('lower penalty\n(less sensitive)',1685,261,159,48,{size:23,color:C.teal});line(1677,279,1649,276,{color:C.teal,width:2.4});line(1649,276,1621,264,{color:C.teal,arrow:true,width:2.4});
 text('Codebook geometry (equal action error)',1301,351,550,36,{size:28,bold:true,color:C.ink});
 for(let r=0;r<5;r++)for(let c=0;c<5;c++)box(1319+c*43,403+r*40,17,17,'#EDF3F8','#859FB5',1.2);ellipse(1414,491,23,98,{color:'#4398A4',fill:'#EDF7F7',width:1,dash:true});box(1405,483,18,17,'#16868C','#057380',1.1);box(1405,523,18,17,'none',C.teal,1.6);
 box(1554,403,22,23,'#13838B','#066E79',1);text('central codeword  w',1595,403,246,30,{size:24,color:C.gray});box(1554,447,22,23,'#D5E5F1','#82A4C3',1);text('other codewords  c',1595,447,239,30,{size:24,color:C.gray});ellipse(1566,525,16,33,{color:'#3198A8',fill:'#F2FAFB',dash:true});text('(v − w)ᵀM(v − w) = const.\n(vertical-to-horizontal\nradius ratio = 3)',1600,502,252,91,{size:23,color:C.gray});
 box(1298,623,550,159,'#F2F4F8','none',0,true);text('The action-sensitive metric allocates finer\ncodebook resolution along the more sensitive\ndirection (horizontal) and coarser resolution\nalong the less sensitive direction (vertical).',1320,641,509,128,{size:26,color:C.gray});
}
for(const[n,fn,asset]of[[10,method,'imagegen-method-reference.png'],[11,geometry,'imagegen-geometry-reference.png']]){
 const s=p.slides.items[n-1];
 for(const sh of [...s.shapes.items])if(!['Title 1','矩形 2','Slide number'].includes(sh.name))sh.delete();
 for(const im of [...s.images.items])im.delete();
 fn(s);
 for(const q of s.shapes.items.filter(x=>x.name==='Reference line'))q.bringToFront();
 for(const q of s.shapes.items.filter(x=>x.name==='Reference arrowhead'))q.bringToFront();
 for(const q of s.shapes.items.filter(x=>x.name==='Reference point'))q.bringToFront();
 const no=s.shapes.items.find(q=>q.name==='Slide number');if(no){no.text=String(n);no.position={left:923,top:501,width:25,height:10};no.text.style={fontSize:9,typeface:'Arial',color:'#697386',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};}
 const scientific=n===10?'Row-vector convention: Y=XWᵀ. With D invertible and HHᵀ=I, X′=XD⁻¹H and W′=WDH preserve the unquantized linear output. The displayed Linear output is an intermediate feature unless the remaining ActionDiT/denoising steps are also included. Final-action sensitivity must propagate through those steps.':'Illustrative values: w=(0,0), c₁=(0.3,0), c₂=(0,0.6). Euclidean squared distances are 0.09 and 0.36. For M=diag(9,1), weighted squared distances are 0.81 and 0.36. The reference L₂ contour label 0.6 is inconsistent with the circle through c₁, whose radius is 0.3. A weighted level-0.36 ellipse has radii 0.2 and 0.6. The geometry is explanatory, not measured experimental data.';
 s.speakerNotes.textFrame.setText(scientific+'\nFigure source: '+path.join(ASSETS,asset)+'.');
}
await(await PresentationFile.exportPptx(p)).save(path.join(TMP,'authored.pptx'));
await fs.mkdir(path.join(TMP,'preview'),{recursive:true});
for(const n of [10,11]){const png=await p.export({slide:p.slides.items[n-1],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'preview',`slide-${n}.png`),Buffer.from(await png.arrayBuffer()));}
console.log(JSON.stringify({slides:p.slides.items.length,authored:path.join(TMP,'authored.pptx')}));
