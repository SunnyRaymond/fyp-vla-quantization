import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/runtime_helpers.mjs';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const ROOT='D:/Downloads/Final Year Project';
const TMP=ROOT+'/.codex-tmp/meeting-20261021';
const ASSET=ROOT+'/Weekly Slides/2026-10-21_assets';
const SOURCE=ROOT+'/Weekly Slides/2026-10-21_Meeting.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const original=[...p.slides.items], bodySource=original[5], coverSource=original[0], endSource=original[11];
const results=JSON.parse(await fs.readFile(ASSET+'/fastwam_results.json','utf8'));
const cosmos=JSON.parse(await fs.readFile(ASSET+'/cosmos_loss.json','utf8'));
const C={blue:'#4C88DE',navy:'#1B2059',ink:'#192C45',gray:'#647184',line:'#CFD9E3',teal:'#087F8C',tealFill:'#E5F5F4',gold:'#BA7A13',goldFill:'#FFF2D6',red:'#B5444F',redFill:'#FBECEE',pale:'#F1F5FA'};
const slides=[], diagrams=new Map(), tableOwners=[];
let activeDiagram=null;
const escape=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function record(type,v){if(activeDiagram)activeDiagram.push({type,...v});}
function text(s,content,x,y,w,h,{size=24,bold=false,color=C.ink,align='left',middle=false,name=''}={}){
 const sh=s.shapes.add({name:name||content.slice(0,55),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=content.split('\n').map(v=>({runs:[v],spaceBefore:0,spaceAfter:0})); sh.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:align,verticalAlignment:middle?'middle':'top',autoFit:'none',wrap:'square',lineSpacing:1.04,insets:{top:0,bottom:0,left:0,right:0}};
 record('text',{text:content,x,y,w,h,size,bold,color,align});return sh;
}
function box(s,content,x,y,w,h,{fill=C.pale,line=C.line,color=C.ink,size=21,bold=false,radius=0}={}){
 const sh=s.shapes.add({name:content||'diagram block',geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1.2},...(radius?{borderRadius:radius}:{})});
 sh.text=content.split('\n').map(v=>({runs:[v],spaceBefore:0,spaceAfter:0}));sh.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'square',lineSpacing:1.04,insets:{left:6,right:6,top:3,bottom:3}};
 record('box',{text:content,x,y,w,h,size,bold,color,fill,line,radius});return sh;
}
function arrow(s,x1,y1,x2,y2,{color=C.ink,width=1.8,dash=false,end=true}={}){
 const sh=s.shapes.add({name:'diagram connector',geometry:'connector',kind:'straight',position:{left:Math.min(x1,x2),top:Math.min(y1,y2),width:Math.abs(x2-x1),height:Math.abs(y2-y1),horizontalFlip:x2<x1,verticalFlip:y2<y1},line:{fill:color,width,style:dash?'dashed':'solid'},...(end?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});
 record('arrow',{x1,y1,x2,y2,color,width,dash,end});return sh;
}
function tokens(s,x,y,n,w,h,fill,line=C.line,gap=4){for(let j=0;j<n;j++)box(s,'',x+j*(w+gap),y,w,h,{fill,line});}
function note(s,str,y=474,size=15.3){text(s,str,64,y,832,33,{size,color:C.gray});}
function heading(s,str,x,y,w=832,size=24,color=C.ink){return text(s,str,x,y,w,35,{size,bold:true,color});}
function body(title,notes='',diagram=false){
 const s=bodySource.duplicate();
 for(const x of [...s.shapes.items])if(x.name!=='Title 1'&&x.name!=='矩形 2')x.delete();
 for(const x of [...s.images.items])x.delete();for(const x of [...s.tables.items])x.delete();
 const t=s.shapes.items.find(x=>x.name==='Title 1');t.text=title;t.position={left:48,top:8,width:880,height:58};
 t.text.style={typeface:'Arial',fontSize:title.length>48?32:112/3,bold:true,color:C.blue,autoFit:'none',wrap:'square',insets:0};
 s.speakerNotes.textFrame.setText(notes);slides.push(s);
 activeDiagram=diagram?[]:null;if(diagram)diagrams.set(title,activeDiagram);
 const tmp=activeDiagram;activeDiagram=null;text(s,String(slides.length),915,483,22,21,{size:12.5,color:C.gray,align:'right'});activeDiagram=tmp;
 return s;
}
function table(s,values,x,y,w,h,widths,{size=19,headerSize=size,highlights=[]}={}){
 const compact=values.map(row=>row.map(v=>String(v).split('\n').map(line=>({runs:[line],spaceBefore:0,spaceAfter:0}))));
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values:compact});
 t.borders.assign({fill:C.line,width:.7,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:size,color:C.ink},margins:{left:8,right:6,top:5,bottom:5},anchor:'center'});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.navy,textStyle:{typeface:'Arial',fontSize:headerSize,bold:true,color:'#FFFFFF'}});
 for(const r of highlights)t.cells.block({row:r,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:C.tealFill,textStyle:{typeface:'Arial',fontSize:size,bold:true,color:C.teal}});
 if(!tableOwners.includes(slides.length))tableOwners.push(slides.length);return t;
}
function lineItem(s,n,title,detail,y,{size=23}={}){
 text(s,String(n).padStart(2,'0'),64,y,44,39,{size:29,bold:true,color:C.teal});
 text(s,title,128,y,768,35,{size,bold:true});text(s,detail,128,y+37,768,56,{size:20,color:C.gray});
}
const fyp=x=>ROOT+'/experiment/idea-validation/'+x;
const src=(...v)=>'\nSources:\n'+v.join('\n');

// 1 — keep the supplied NTU cover, replacing only its text.
activeDiagram=null;
let s=coverSource.duplicate();slides.push(s);
for(const sh of [...s.shapes.items]){
 if(sh.name==='Title 1'){sh.text='Latent + Action-Aware';sh.position={left:289.42,top:184,width:625,height:56};sh.text.style={typeface:'Arial',fontSize:40,bold:true,color:'#FFFFFF',autoFit:'none',wrap:'square',insets:0};}
 if(sh.name==='Meeting subtitle'){sh.text='Microscaling VQ, activation refinement & relational KL';sh.position={left:289.42,top:319,width:625,height:64};sh.text.style={typeface:'Arial',fontSize:23,bold:true,color:'#FFFFFF',autoFit:'none',wrap:'square',insets:0};}
 if(sh.name==='Subtitle 2'){sh.text='Guo Yichen\nCollege of Computing and Data Science (CCDS)\n21 October 2026';sh.text.style={typeface:'Arial',fontSize:20,bold:true,color:'#FFFFFF',autoFit:'none',insets:0};}
}
text(s,'WAM Quantization',289.42,241,625,57,{size:40,bold:true,color:'#FFFFFF'});
s.speakerNotes.textFrame.setText('本次汇报顺序：已实现方法 → 所有有代表性的本地 LIBERO+ 闭环结果 → future intervention → 旧 loss 与闭环的差距 → Soft-bin + relational KL → potential contributions 与下一步。所有闭环数据来自已完成本地报告，截止 20 October 2026。');

// 2 — editable teacher/student method, rebuilt from the generated reference.
s=body('Latent + action-aware Microscaling VQ',
'已实现的 Fast-WAM weight quantization 框架。Teacher 和 student 共享 observations、language、proprioception、sampler seeds。Student 通过自身生成的 future 构建 video K/V；不注入 teacher future 训练。所有 614 target Linear 覆盖 video/action/proprio 等目标模块。Weight calibration 更新 codebooks 和 row gains；indices、B32 exponents 和 transforms 固定。图为结构示意，tokens 数量不代表真实 tensor 大小。'+src(fyp('fastwam-vq-c8-action-latent-plus/README.zh.md'),fyp('fastwam-vq-action-latent/README.zh.md'),ASSET+'/method-imagegen-reference.png'),true);
text(s,'Observation + language + proprioception',64,82,832,30,{size:22,bold:true});
box(s,'Shared\ninputs',64,179,100,99,{fill:C.pale,size:21});
heading(s,'BF16 teacher · frozen',205,123,450,23,C.navy);
const ty=164,sy=285;
box(s,'Video DiT',205,ty,148,63,{fill:'#E8F1FC',line:'#7EACDC',bold:true});
box(s,'Future Zₜ',392,ty,128,63,{fill:C.goldFill,line:'#DCBB74',bold:true});
box(s,'Action DiT',559,ty,148,63,{fill:'#E8F1FC',line:'#7EACDC',bold:true});
box(s,'Actions aₜ',746,ty,143,63,{fill:C.tealFill,line:'#73B8B7',bold:true});
heading(s,'Quantized WAM',205,245,241,23,C.red);
box(s,'Video DiT\nVQ + A4',205,sy,148,69,{fill:C.redFill,line:'#D89BA2',bold:true});
box(s,'Own future Zₛ',392,sy,128,69,{fill:C.goldFill,line:'#DCBB74',bold:true});
box(s,'Action DiT\nVQ + A4',559,sy,148,69,{fill:C.redFill,line:'#D89BA2',bold:true});
box(s,'Actions aₛ',746,sy,143,69,{fill:C.tealFill,line:'#73B8B7',bold:true});
for(const y of [ty+32,sy+34]){arrow(s,353,y,392,y);arrow(s,520,y,559,y);arrow(s,707,y,746,y);}
arrow(s,164,208,184,208,{end:false});arrow(s,184,196,205,196);arrow(s,184,319,205,319);arrow(s,184,196,184,319,{end:false});
text(s,'video K/V',516,229,90,24,{size:15,color:C.gray});
arrow(s,455,227,455,285,{color:C.gold,dash:true,end:false});text(s,'Latent loss',378,363,160,26,{size:20,bold:true,color:C.gold,align:'center'});
arrow(s,817,227,817,285,{color:C.teal,dash:true,end:false});text(s,'Action loss',741,363,154,26,{size:20,bold:true,color:C.teal,align:'center'});
text(s,'MSE(first 10 × 6 motor actions) + ρc · NMSE(future latents)',100,409,780,35,{size:25,bold:true,align:'center'});
note(s,'614 target Linears; update codebooks + row gains. Latents exclude frame 0; motor loss excludes gripper.',469,16);

// 3 — editable encoding diagram and exact bit arithmetic.
s=body('Microscaling additive VQ weights',
'每一 target Linear 共享两个 K=256,d=4 BF16 codebooks。每四个 scalar weight 使用两条 uint8 indices，core cost 为 16/4=4 bpw。每 output row 保留 BF16 RMS row gain r；沿 input dimension 每 32 个 weights 用一个 uint8 exponent，贡献 8/32=0.25 bpw。e_g=round(log2(block_RMS/r_row))，拟合中固定。有效预算含 codebooks、gains、tails 和 transforms，Fast-WAM row-scale≈4.020873，microscaling≈4.270878。Weight calibration 用 FP32 masters/STE 调整 BF16-stored books 与 row gains。'+src(fyp('fastwam-vq-microscale/README.zh.md'),fyp('fastwam-vq-microscale/RESULTS.zh.md'),fyp('fastwam-vq-c8-action-latent-plus/README.zh.md')),true);
heading(s,'Transformed weight row',64,88,400,23);
for(let g=0;g<3;g++){tokens(s,64+g*275,133,8,28,27,g===0?C.goldFill:'#EBF2FA','#A9B8C8',3);text(s,g===2?'…':'B32 = 8 vectors × d4',64+g*275,167,245,28,{size:18,color:C.gray,align:'center'});}
text(s,'i₁  (8-bit)',78,226,125,29,{size:21,bold:true,color:C.teal});text(s,'i₂  (8-bit)',78,316,125,29,{size:21,bold:true,color:C.gold});
box(s,'Codebook C₁\n256 × 4 · BF16',237,211,233,67,{fill:C.tealFill,line:'#72B4B5',size:23,bold:true});
box(s,'Codebook C₂\n256 × 4 · BF16',237,301,233,67,{fill:C.goldFill,line:'#D9B466',size:23,bold:true});
arrow(s,196,243,237,243);arrow(s,196,334,237,334);
arrow(s,470,244,532,244,{end:false});arrow(s,532,244,532,290,{end:false});
arrow(s,470,335,532,335,{end:false});arrow(s,532,335,532,290,{end:false});
box(s,'+',514,270,37,40,{fill:'#FFFFFF',line:C.navy,size:28,bold:true});arrow(s,552,290,595,290);
box(s,'× r_row × 2^(e_g)',595,254,294,74,{fill:C.pale,line:'#93A9C2',size:24,bold:true});
text(s,'e_g = round(log₂(block_RMS / r_row))',518,214,380,29,{size:18,color:C.gray});
text(s,'BF16 row gain + B32 exponent',585,343,320,28,{size:19,color:C.gray,align:'center'});
heading(s,'4 core bpw  +  0.25 B32 bpw  +  codebooks / gains / transforms  ≈  4.27 bpw',64,404,832,22,C.teal);
note(s,'Fixed indices + exponents; calibrate codebooks + row gains. VQ codewords remain BF16 values.',471,16);

// 4 — editable activation path; per-channel D differs from per-layer r.
s=body('Per-channel scaling + activation clipping',
'这是 per-input-channel scaling + per-Linear activation clipping，不是每 channel 一个 clipping threshold。D=2^log2_D 为每个 physical input channel 一个正的 continuous scale；r_l 为每 Linear 一个 ratio，跨 token/denoising steps 共享。先对 input 施加已冻结 Smooth0.5/H128，然后 v=z/D；每 token s=r_l max|v|/7，signed A4 q=clamp(round(v/s),-7,7)。decoded BF16 Wq 乘相同 D 后以 BF16 缓存。忽略 rounding 时 compensating Linear 恒等；已 quantized 情况下改变 error。C80 refinement 从 C8 rho1.5 开始，80 updates，只训 D/r_l，VQ 参数冻结。使用 coupled dynamic-absmax surrogate 和 STE。'+src(fyp('fastwam-vq-per-channel-scaling-clipping/channel_banks.py'),fyp('fastwam-vq-c80-per-channel-plus/README.zh.md')),true);
heading(s,'Activation path',64,89,420,23,C.teal);
box(s,'X',64,143,60,66,{fill:'#FFFFFF',size:26});
box(s,'Smooth 0.5\n+ H128',153,143,156,66,{fill:'#E8F1FC',line:'#7EACDC',bold:true});
box(s,'÷ D',339,143,105,66,{fill:C.goldFill,line:'#D9B466',size:26,bold:true});
box(s,'Dynamic\nsigned A4',474,143,164,66,{fill:C.tealFill,line:'#72B4B5',bold:true});
box(s,'BF16\ndense Linear',704,143,185,66,{fill:C.pale,line:'#93A9C2',bold:true});
for(const [a,b] of [[124,153],[309,339],[444,474],[638,704]])arrow(s,a,176,b,176);
text(s,'z',312,151,24,25,{size:20});text(s,'v',447,151,24,25,{size:20});
heading(s,'Weight path',64,255,420,23,C.gold);
box(s,'Frozen VQ\nrepresentation',64,297,210,61,{fill:C.goldFill,line:'#D9B466',bold:true});
box(s,'Decode Wq',313,297,163,61,{fill:C.goldFill,line:'#D9B466',bold:true});
box(s,'BF16(Wq · D)\ncache once',515,297,185,61,{fill:C.pale,line:'#93A9C2',size:22,bold:true});
arrow(s,274,328,313,328);arrow(s,476,328,515,328);arrow(s,700,328,795,328,{end:false});arrow(s,795,328,795,209);
text(s,'s_t = r_l · max |v_t| / 7',66,398,405,34,{size:25,bold:true});
text(s,'q = clamp(round(v / s_t), −7, 7)',475,398,421,34,{size:24,bold:true});
note(s,'Learn D per channel + r_l per Linear on C80; keep VQ weights fixed. BF16 KV; native VQ kernel pending.',467,16);

// 5/6 — all completed Fast-WAM LIBERO+ configurations.
const arms=results.closed_loop_results;
const short=['BF16','QuaRot adapted','SpinQuant adapted','Smooth + H128 Scalar','Row-scale VQ + A4','B32 Micro VQ + A4','C8 A+Z, ρ0.1 · 2 epochs','C8 A+Z, ρ1.5 · 1 epoch','C80 clip-only','C80 joint VQ + clip','C80 channel scales + clip'];
const bpw=['16.00','≈4.13†','≈4.13†','4.14','4.02','4.27','4.27','4.27','4.27','4.27','4.28'];
function resultRows(indices){return [['Method','Eq. bpw','Spatial','Object','Goal','Long','Overall'],...indices.map(i=>{const a=arms[i],u=a.suite_successes;return [short[i],bpw[i],String(u.libero_spatial.successes),String(u.libero_object.successes),String(u.libero_goal.successes),String(u.libero_10.successes),`${a.total.successes} (${a.total.success_rate_pct.toFixed(2)}%)`];})];}
const mainSources=[fyp('fastwam-vq-plus/results/RESULTS.zh.md'),fyp('fastwam-rotation-baselines/protocol.json'),fyp('fastwam-smooth-scalar-plus/source_receipts/receipt.json')];
s=body('Fast-WAM: LIBERO+ local baselines',
'所有 rows 都来自本地完成的 frozen 280-variant cohort，不是 paper-reported numbers。每 suite 分母 70。QuaRot/SpinQuant 是针对 Fast-WAM 的本地 adaptation，不等同官方完整 recipe。Scalar 与 rotation baselines 实际采用 native packed S4 path，而 VQ 是 decode BF16；本表只比较 closed-loop success 与编码预算，不比较 latency。不同 arm calibration 覆盖和目标不同，不是单因素消融。Rotation bpw 的4.125是4bit code+BF16 scale/128 的估算，未把 rotations/tails 算入，因此加 dagger，避免误报为实测全 overhead payload。'+src(...mainSources));
text(s,'Frozen paired cohort: 4 suites × 7 perturbations × 10 variants = 280 episodes / arm',64,83,832,43,{size:20});
table(s,resultRows([0,1,2,3,4,5]),64,142,832,274,[246,76,85,85,85,85,170],{size:18.5,headerSize:18,highlights:[6]});
text(s,'Microscaling VQ: 204/280 vs local SpinQuant 203/280',64,435,832,28,{size:23,bold:true,color:C.teal});
note(s,'Overall denominator: 280; suite cells: /70. Low-bit arms: W4A4. † bpw estimates exclude rotation/tail overhead.',474,14.3);

s=body('Fast-WAM: supervision and refinement',
'同一 frozen 280-episode cohort 的全部后续完成配置。C8表示weight calibration contexts；C80表示 activation-refinement contexts，不是 rollout 数。ρ0.1 完整2epochs/16updates，ρ1.5为1epoch/8updates；这是各自冻结配置，不是matched training budget消融。C80 clip-only207，joint202，per-channel203；joint per-channel虽有fixed-input结果尚未发现完成闭环，不能加入completed表。该cohort未有独立seed重复，不应对1—5个success小差异宣称稳定统计优势。'+src(fyp('fastwam-vq-c8-action-latent-plus/results/RESULTS.zh.md'),fyp('fastwam-vq-c80-clip-joint-plus/RESULTS.zh.md'),fyp('fastwam-vq-c80-per-channel-plus/RESULTS.zh.md')));
text(s,'A+Z: action+latent · C8 weight calibration / C80 activation refinement',64,84,832,34,{size:21});
table(s,resultRows([0,5,6,7,8,9,10]),64,134,832,300,[246,76,85,85,85,85,170],{size:18,headerSize:18,highlights:[5]});
text(s,'Best completed low-bit result here: clip-only 207/280 (73.93%)',64,450,832,30,{size:22,bold:true,color:C.teal});
note(s,'Overall: /280; suite: /70. Target-weight bpw includes metadata; VQ uses decoded BF16 + fake A4.',482,14);

// 7 — independent Cosmos protocol, model result not mixed with Fast-WAM.
s=body('Cosmos Policy: LIBERO+ transfer',
'Cosmos-Predict2-2B direct policy，planning disabled。三臂840episodes全部完成，每臂280。5denoising steps、16action chunk并执行16steps；与Fast-WAM10/10和执行10steps不同，不能跨model数值比较来归因量化方法。C8共享observations但teacher由Cosmos自身产生。VQ学习codebooks/rowgains，目标为16×6motor MSE+future-image slots6/7 NMSE。SpinQuant200updates，VQ16updates且objectives不同，因此不是action/latent loss贡献的单因素消融。有效bpw均含编码开销；whole DiT bpw另计未量化参数。'+src(fyp('cosmos-policy-c8-w4a4/RESULTS.zh.md'),fyp('cosmos-policy-c8-w4a4/retrieved/results/evaluation/summary.json')));
text(s,'Direct policy · C8 calibration · same 280 paired variants per arm',64,85,832,34,{size:21});
table(s,[['Method','Eq. bpw','Spatial','Object','Goal','Long','Overall'],['BF16','16.00','59','62','54','58','233 (83.21%)'],['SpinQuant adapted W4A4','4.13','54','61','46','60','221 (78.93%)'],['Action+latent Micro VQ','4.29','57','62','49','58','226 (80.71%)']],64,140,832,186,[246,76,85,85,85,85,170],{size:18.5,headerSize:18,highlights:[3]});
heading(s,'+5 successes vs SpinQuant; −7 vs BF16',64,353,832,26,C.teal);
text(s,'VQ uses 3.92% more target-weight bits than SpinQuant.\nWhole-DiT bpw: 5.45 vs 5.31; transferability of the current recipe is demonstrated.',64,397,832,66,{size:21});
note(s,'Overall: /280; suite: /70. bpw covers target weights. Decoded BF16 / fake A4; native speedup remains untested.',478,15);

// 8 — actual fixed-reader intervention, not the typo in the weekly prose.
s=body('Does protecting the future help downstream actions?',
'干预冻结 downstream BF16 reader，包括 video prepare/prefill、action DiT/proprio 等，注入不同 donor 的 final future，重新构建 K/V。完整60contexts包括8cal、4selection×2seed、22test×2seed；表格是22test cases先平均2seeds的case mean。BF16 donor自重放误差0，Parent=0.036144691，Action-only=.041243219，Action+latent=.035083670。13/22first10改善，full32/late22各14/22。是已见cohort的fixed-input intervention，0 executed episodes，不是真实future accuracy或新泛化证据。'+src(fyp('fastwam-vq-future-reader/retrieved/results/analysis_teacher/REPORT.zh.md')),true);
box(s,'Future donor\nParent / action-only / +latent',64,102,318,73,{fill:C.goldFill,line:'#D9B466',size:21,bold:true});
box(s,'Fixed BF16 reader\nvideo K/V → Action DiT',449,102,318,73,{fill:'#E8F1FC',line:'#7EACDC',size:22,bold:true});
arrow(s,382,139,449,139);arrow(s,767,139,818,139);box(s,'a',818,111,70,55,{fill:C.tealFill,line:'#72B4B5',size:28,bold:true});
activeDiagram=null;
table(s,[['Future donor','First-10 motor RMSE ↓','Vs action-only'],['Parent','0.036145','−12.36%'],['Action-only','0.041243','Reference'],['Action+latent','0.035084','−14.93%']],64,211,832,178,[310,270,252],{size:22,highlights:[3]});
heading(s,'Protecting future latents improves mean downstream motor fidelity',64,413,832,23,C.teal);
note(s,'Earlier row-scale VQ family: 22 cases × 2 seeds; 13/22 improve. Fixed-input teacher fidelity; no closed-loop claim.',470,15);

// 9 — cross-reader signal and matched-energy limitation, kept together.
s=body('Future protection across readers',
'左表来自2×2cross-reader intervention，reader固定action-only或action+latent时替换future donor。这里reader不再是BF16。Action-only reader mean .058110442→.053481176，action+latent reader .058676505→.052951009。分别10/22、12/22cases改善，平均收益不等于每case收益。右表是另一个fixedBF16reader NMSE-matched control：C=T+alpha(A−T)，经有界BF16correction逐context匹配B的NMSE，alpha不使用action选值。B−C mean−.00099049但median+.000042169，11/22改善，leave-one-out可变正，full32/late22比C差。原均值收益约84%由总future误差幅度降低重现。不能宣称已确认特定semantic/spatial有害drift。'+src(fyp('fastwam-vq-future-reader/retrieved/results/analysis_cross/REPORT.zh.md'),fyp('fastwam-vq-future-nmse-matched/RESULTS.zh.md')));
heading(s,'The mean gain transfers across readers',64,92,832,22,C.teal);
table(s,[['Fixed student reader','A-only future','A+latent future'],['Action-only','0.058110','0.053481'],['Action+latent','0.058677','0.052951']],64,140,832,124,[340,246,246],{size:21});
heading(s,'Most of the gain can be reproduced by reducing future error energy',64,291,832,22,C.gold);
table(s,[['Future donor · fixed BF16 reader','Future NMSE ↓','Motor RMSE ↓'],['A: action-only','0.093011','0.041243'],['C: A, NMSE matched to B','0.071786','0.036074'],['B: action+latent','0.071787','0.035084']],64,334,832,137,[400,216,216],{size:19,highlights:[3]});
note(s,'B vs C improves 11/22 cases; no robust extra benefit at matched NMSE. This supports latent fidelity, not a unique mechanism.',484,14.4);

// 10 — original loss and direct mismatch to actual environment success.
s=body('MSE fidelity and closed-loop success',
'原训练loss是first10motor MSE+rho*c*futureNMSE，c=0.028303017887797623，rho1.5，lambda=.04245452683169643。RMSE仅用来报告，优化的是MSE。c依据初始化calibrationmean冻结，rho不是gradient ratio。Fixedinput来自40heldoutcases×2seeds，四suite等权。C80clip .0438484373/.0548481241，joint .0418468372/.0489733362，perchannel .031641466/.055377875；对应闭环207/202/203。Heldout metric和closedloop是不同测量而且访问状态分布不同。'+src(fyp('fastwam-vq-per-channel-scaling-clipping/results/RESULTS.zh.md'),fyp('fastwam-vq-c80-clip-joint-plus/RESULTS.zh.md'),fyp('fastwam-vq-c80-per-channel-plus/RESULTS.zh.md')));
text(s,'Original objective:  L = MSE_action + ρc · NMSE_future',64,87,832,39,{size:26,bold:true});
text(s,'c is frozen from initial calibration loss values; ρ = 1.5',64,131,832,30,{size:20,color:C.gray});
table(s,[['C80 refinement','Motor RMSE ↓','Latent NMSE ↓','LIBERO+ success ↑'],['Clip-only','0.043848','0.054848','207/280 · 73.93%'],['Joint VQ + clipping','0.041847','0.048973','202/280 · 72.14%'],['Per-channel + clipping','0.031641','0.055378','203/280 · 72.50%']],64,195,832,178,[300,160,166,206],{size:21,highlights:[3]});
heading(s,'−27.84% motor RMSE, but −4 closed-loop successes vs clip-only',64,403,832,23,C.red);
text(s,'Motivation: preserve the structure of action sequences and future latents,\nthen test whether the objective better tracks rollout failures.',64,444,832,52,{size:21});

// 11 — distribution construction; no histogram pooling or invented Gaussian center softmax.
s=body('Soft-bin KL: preserve absolute values',
'真实实现softbin_loss.py以Gaussian CDFdifference而不是centroidsoftmax生成每scalar16bin分布。z=(x−mu_c)/max(std_c,.02)，mu/std仅完整C80teacher计算并冻结。15interioredges[-3.5,3.5]等间距，sigma=.375，两个无限tail。每一对应scalar位置算KL(teacher||student)再平均，不是整条actiontrajectory先pool成histogram。Teacher detached。alpha_a=.01230247928,alpha_z=.31297895225，初始C80 MSE/KL和NMSE/KL配平，保留原rho*c。只训D/r_l，80updates，固定final80；books/gains/indices/exponents不变。'+src(fyp('fastwam-vq-dual-kl/softbin_loss.py'),fyp('fastwam-vq-dual-kl/results/prepare/loss_scaling.json')),true);
box(s,'Teacher xₜ',64,119,164,60,{fill:'#E8F1FC',line:'#7EACDC',size:23,bold:true});
box(s,'Student xₛ',64,242,164,60,{fill:C.redFill,line:'#D89BA2',size:23,bold:true});
box(s,'Shared frozen\nchannel normalization',291,147,255,102,{fill:C.pale,line:'#93A9C2',size:23,bold:true});
arrow(s,228,149,267,149,{end:false});arrow(s,267,149,267,178,{end:false});arrow(s,267,178,291,178);
arrow(s,228,273,267,273,{end:false});arrow(s,267,273,267,219,{end:false});arrow(s,267,219,291,219);
text(s,'z = (x − μ_c) / max(σ_c, 0.02)',284,266,282,52,{size:21,align:'center'});
arrow(s,546,198,597,198);
box(s,'16-bin distribution\nGaussian CDF mass\nσ = 0.375',597,145,292,108,{fill:C.tealFill,line:'#72B4B5',size:23,bold:true});
tokens(s,602,273,16,13,24,C.tealFill,'#72B4B5',5);
text(s,'15 interior edges: −3.5 … 3.5 + 2 tails',584,310,313,50,{size:18,color:C.gray,align:'center'});
text(s,'L_bin = mean over scalar positions KL(p_teacher || p_student)',64,372,832,36,{size:25,bold:true,color:C.teal,align:'center'});
text(s,'Apply separately to first 10 × 6 motor values and future latent values.\nSoft-bin keeps the absolute-value anchor when relation distances have blind spots.',64,426,832,67,{size:21});

// 12 — all token/neighborhood design details as editable objects.
s=body('Temporal / spatial relational KL',
'关系代码用每feature mean-squared pairwise distance，不是Gram matrix。Action标准化first10×6，经length3stride1 orderedwindows得到8×18tokens，全部非selfwindows相邻。Latentshape[1,48,3,14,28]去frame0，按widthsplit两camera各14×14，各自2×2avgpool到7×7，2times×2views×49=196token×48features。邻边是samecamera,sametime的8neighbors以及sameposition跨time。Boundary spatial节点少于8邻居。无跨cameraedge，无action-latent强制时间插值。Teacher概率detach，tau依据C80teacherpositiveedgedistancemedian冻结。Softbin检测关系对整体translation/signflip等的盲区。'+src(fyp('fastwam-vq-relational-kl/relational_loss.py'),fyp('fastwam-vq-relational-kl/README.zh.md')),true);
heading(s,'Action: ordered temporal windows',64,88,416,23,C.teal);
tokens(s,64,134,10,30,29,C.tealFill,'#72B4B5',5);
text(s,'10 steps × 6 motor dimensions',64,173,371,26,{size:19,color:C.gray});
arrow(s,243,204,243,235,{color:C.teal});
box(s,'[a₁,a₂,a₃]   [a₂,a₃,a₄]  …\n8 ordered tokens × 18 features',64,244,366,73,{fill:C.tealFill,line:'#72B4B5',size:22,bold:true});
text(s,'Neighbors: all other windows\nSelf-pairs masked',64,334,366,62,{size:22});
heading(s,'Latent: spatial + temporal graph',488,88,408,23,C.gold);
text(s,'2 future times × 2 views × 7 × 7\n196 tokens × 48 features',488,125,408,58,{size:22,bold:true});
const cell=20,gap=2;
for(let t=0;t<2;t++){
 const x=490+t*225,y=211;
 for(let r=0;r<7;r++)for(let c=0;c<7;c++)box(s,'',x+c*(cell+gap),y+r*(cell+gap),cell,cell,{fill:(r===3&&c===3)?'#EEA454':(t===0&&Math.max(Math.abs(r-3),Math.abs(c-3))===1)?'#F9D592':C.goldFill,line:'#E5C785'});
 text(s,'time '+(t+1),x,183,154,25,{size:19,color:C.gold,align:'center'});
}
for(let r=-1;r<=1;r++)for(let c=-1;c<=1;c++)if(r||c)arrow(s,566,287,566+c*22,287+r*22,{color:C.gold,width:1,end:false});
arrow(s,577,286,789,286,{color:C.red,dash:true});
text(s,'8 spatial neighbors + same-position time edge\n2×2 pool per view; no cross-camera edges',488,381,408,60,{size:19});
text(s,'dᵢⱼ = mean (uᵢ − uⱼ)²    p(j|i) = softmax(−dᵢⱼ / τ)',64,450,832,35,{size:24,bold:true,align:'center'});
note(s,'Match teacher → student neighbor distributions with KL; frozen teacher statistics and temperature.',486,14.5);

// 13 — composite objective with its fitted coefficients, no rollout prediction claim yet.
s=body('The combined KL objective',
'L=alpha_a*(action_KL+beta_a*action_temporal_KL)+rho*c*alpha_z*(latent_KL+beta_z*latent_spatiotemporal_KL)。Teacheronlycalibrationfitting：alpha用C80initMSE/KL与NMSE/KL；tau_a=.05844502896,tau_z=.42243281007；beta_a=.61968510916,beta_z=27.14823378066，通过预设standardizednoise std .1的0.25relativeweight配平。0.25并不意味着真实modelerror或gradient中占25%。D/r_l独立从C8rho1.5checkpoint初始化，各80updates，未用heldoutselectfinal80。Distance关系本身有translation/signflip/部分timereversal盲区，softbin绝对anchor补足，不能声称所有结构错误可识别。'+src(fyp('fastwam-vq-relational-kl/results/prepare/summary.json'),fyp('fastwam-vq-relational-kl/protocol.json'),fyp('fastwam-vq-relational-kl/results/ANALYSIS.zh.md')));
text(s,'L = α_a [ L_bin,a + β_a L_temporal,a ]',64,104,832,49,{size:31,bold:true,color:C.teal});
text(s,'     + λ α_z [ L_bin,z + β_z L_spatiotemporal,z ]',64,158,832,49,{size:31,bold:true,color:C.gold});
table(s,[['Component','Definition / calibration rule'],['α_a, α_z','Initial C80 MSE / soft-bin KL; NMSE / soft-bin KL'],['λ = ρc','Original action–latent balance; ρ=1.5, λ≈0.04245'],['τ_a, τ_z','Median positive C80 teacher neighbor distance'],['β_a, β_z','0.25 × soft-bin KL / relation KL under fixed teacher noise']],64,238,832,185,[200,632],{size:20});
text(s,'Only student D and clipping ratios are updated.\nAll statistics and weights are frozen before training; teacher is detached.',64,448,832,52,{size:21,bold:true});

// 14 — actual KL endpoint numbers and honest ongoing rollout status.
s=body('KL refinement: current evidence',
'本表来自原40heldoutcase×2seeds，先seedcase平均再四suite等权。Softbinrelationaltrain各1trajectory、80updates，固定final80，不在heldout选ckpt。原MSEfinal80是历史同协议endpoint重放比较。Softbin.03232986/.053033005，relational.031438487/.054295895；temporal .040418109→.034484781；spatiotemporal .0063970953→.006147169；composite .0059133057→.0057434071。KL闭环评测在最近local快照431/560episode记录，未有completecriterion。进度不是success，pending不能替代最终rates。'+src(fyp('fastwam-vq-relational-kl/results/ANALYSIS.zh.md'),fyp('fastwam-vq-relational-kl/results/aggregate/summary.json'),ASSET+'/cosmos_loss.json'));
text(s,'Same held-out 40 cases × 2 seeds · fixed final80 checkpoints',64,84,832,34,{size:21});
table(s,[['Refinement loss','Motor RMSE ↓','Latent NMSE ↓','Temporal KL ↓','Latent relation KL ↓'],['MSE + NMSE (historical)','0.031641','0.055378','—','—'],['Soft-bin','0.032330','0.053033','0.040418','0.006397'],['Soft-bin + relational','0.031438','0.054296','0.034485','0.006147']],64,143,832,180,[276,135,135,135,151],{size:18.5,headerSize:17.7,highlights:[3]});
text(s,'Adding relations: temporal KL −14.68%, latent relation KL −3.91%\nMotor RMSE −2.76%, while latent NMSE rises 2.38% vs soft-bin',64,351,832,63,{size:22,bold:true,color:C.teal});
text(s,'Paired LIBERO+ evaluation is in progress; final success rates are pending.',64,442,832,55,{size:23,bold:true,color:C.red});

// 15 — proposed contribution language, status explicit.
s=body('Potential contributions',
'用户的四点contribution按证据强度润色：1当前fixedinputfutureintervention支持latentfidelity的重要性，但likelyvulnerable机制还待验证；2loss已实现，forecastrollouterror仍为researchgoal，当前没有证明predictiveassociation；3Fast-WAM+Cosmos框架/闭环已实现，similarbpw下cohortcurrentrecipecompetitive，但first需priorartcheck且不声称SOTA或统计显著；4nativekernel尚未实现。所有措辞作为proposedpapercontributions，不把目标写成completedclaim。');
lineItem(s,1,'Identify future-latent fidelity as a key quantization target','Evidence: fixed-reader interventions link latent preservation to mean action fidelity.',92);
lineItem(s,2,'Develop latent + action relational distillation for WAMs','Aim: preserve value and structure, and better anticipate rollout error.',188);
lineItem(s,3,'Build a VQ framework with competitive performance at similar bpw','Demonstrated on Fast-WAM and Cosmos Policy; “first” requires a prior-art audit.',284);
lineItem(s,4,'Deliver native kernel acceleration for the VQ framework','Planned contribution · not implemented yet; no native speedup claimed.',380);

// 16 — next steps in the user's requested order, no invented date promises.
s=body('Next steps',
'近期优先完成冻结checkpoint的pairedLIBERO+评测；loss优化需以闭环success与失败轨迹为目标，同时保留原cohortcontrols。候选消融：MSEvssoftbin、action-onlyvslatent-onlyvsboth、temporal-onlyvsspatial/temporal-latent、D/r的分离、C8/C80等校准规模，匹配bpw与预算。换baseline：CosmosPolicy on LIBERO/LIBERO+，LingBot-VA2 on RoboTwin2.0，均为planned未实现本轮不保证可行性/日期。真实robot需要确认hardware/tasks，报告success、rollout error与endtoendlatency。Kernel关注decode/transformprofiling、batchedtransform接线、nativeVQ执行与真实latency，不将compressedpayload当作VRAM。CVPR表达愿望无未查证deadline。');
lineItem(s,1,'Refine KL against closed-loop outcomes','Finish the paired run; inspect failure traces and improve loss balance.',86,{size:23});
lineItem(s,2,'Run controlled ablations','Soft-bin / temporal / spatial terms; action–latent balance; D versus clipping.',169,{size:23});
lineItem(s,3,'Expand models and benchmarks','Cosmos Policy on LIBERO / LIBERO+; LingBot-VA2 on RoboTwin 2.0.',252,{size:23});
lineItem(s,4,'Validate on a real robot and implement native acceleration','Measure task success and end-to-end latency; establish the actual deployment gain.',335,{size:23});
text(s,'Hope to catch CVPR :)',128,455,768,38,{size:29,bold:true,color:C.teal});

// 17 — supplied closing layout.
activeDiagram=null;s=endSource.duplicate();slides.push(s);
const endTitle=s.shapes.items.find(x=>x.name==='Title 1');endTitle.text='Questions and Discussion';
s.speakerNotes.textFrame.setText('讨论重点：当前proxy与闭环目标之间的差距、KL结构设计的优先级、跨模型和真机验证范围，以及kernel工作量。');

// 18 — supporting protocol/bpw definitions kept available for questions.
s=body('Appendix: protocol and bit-budget scope',
'FastWAM模型614targetLinears/6016317440weights。encodedbpw统一bytes*8/Ntarget。Scalar receipt包含transforms，VQ包含indices/books/gains/B32exps/transforms；perchannel另计FP32log2D。Rotation只有code+G128scaleestimate4.125，未有统一completepayload。Cosmos同target280Linear1761607680weights。两model不同steps/chunks因此不跨模型比较latency或success来声称量化方法优劣。闭环manifest280是固定subset，不是全LIBEROPlusbenchmark。'+src(...mainSources,fyp('fastwam-vq-c80-per-channel-plus/RESULTS.zh.md'),fyp('cosmos-policy-c8-w4a4/RESULTS.zh.md')));
table(s,[['Condition','Fast-WAM','Cosmos Policy'],['Closed-loop cohort','4 × 7 × 10 = 280 variants / arm','Same 280-variant protocol'],['Sampler / chunk','10 video + 10 action steps; execute 10/32','5 denoising steps; execute 16/16'],['Target weights','614 Linear layers; 6.016B weights','280 Linear layers; 1.762B weights'],['VQ encoding','d4, 2 × K256 BF16 books, B32','Same encoding family'],['Execution','Decoded BF16; fake A4; BF16 KV','Decoded BF16; fake A4; BF16 KV']],64,103,832,280,[190,330,312],{size:18.7,headerSize:20});
text(s,'Equivalent bpw = encoded payload bytes × 8 / target weight count',64,405,832,34,{size:24,bold:true});
text(s,'Includes quantizer metadata where measured; excludes activations, KV and non-target model weights.\nEncoded bpw is not runtime VRAM. Fast-WAM rotation baselines use a different native S4 path.',64,450,832,47,{size:18});

// 19 — full C80 metrics, including the implemented joint per-channel variant without fabricated rollouts.
s=body('Appendix: C80 activation refinements',
'Fixedinputevidence：same40heldoutcases×2seed/equal4suites。Baseline.044296774/.060877371;clip.043848437/.054848124;b32.043986743/.054151555;joint.041846837/.048973336;perchannel.031641466/.055377875;jointperchannel.032668?.048842。Jointperchannel数字从周报四舍五入，完整源results。本表用于记录已实现的可训练参数组合，不给没有完成closedloop的row补成功率。'+src(fyp('fastwam-vq-per-channel-scaling-clipping/results/RESULTS.zh.md'),fyp('fastwam-vq-joint-per-channel/results/RESULTS.zh.md'),'https://app.notion.com/p/3fd44c7da965815ead32f7858e8e98db'));
table(s,[['Configuration','Motor RMSE ↓','Latent NMSE ↓','Closed-loop successes'],['C8 ρ1.5 baseline','0.044297','0.060877','206/280'],['Clip-only','0.043848','0.054848','207/280'],['B32 scaling + clipping','0.043987','0.054152','Not evaluated'],['Joint VQ + clipping','0.041847','0.048973','202/280'],['Per-channel + clipping','0.031641','0.055378','203/280'],['Joint VQ + per-channel','0.032668','0.048842','Not evaluated']],64,111,832,295,[315,155,155,207],{size:20,headerSize:19,highlights:[5]});
text(s,'Per-channel scaling substantially improves motor fidelity.\nJoint optimization and latent fidelity trade-offs still need closed-loop explanation.',64,439,832,60,{size:22});

// 20 — explicit boundary of present execution, with measured cost useful for discussion.
s=body('Appendix: execution cost and kernel work',
'warmfullactionquerycost来自不同rolloutquery流，非fixedinputpairedbenchmark。FastWAMclip4.99sec/joint4.96/perchannel4.90508；CosmosBF16.59399,Spin1.05653,VQ10.04213。CosmosVQ在线blockH128path未接入batchedtransform，即使summary标签写batched也不能声称已生效。此页保持实现scope避免把编码4bpw当成nativeacceleration。'+src(fyp('fastwam-vq-c80-clip-joint-plus/RESULTS.zh.md'),fyp('fastwam-vq-c80-per-channel-plus/RESULTS.zh.md'),fyp('cosmos-policy-c8-w4a4/RESULTS.zh.md')));
table(s,[['Model / implementation','Warm full-query mean','Interpretation'],['Fast-WAM C80 clip / joint','4.99 s / 4.96 s','Decoded BF16 + batched H128 + fake A4'],['Fast-WAM per-channel + clipping','4.91 s','Cached BF16 compensated weights'],['Cosmos BF16 / SpinQuant / VQ','0.594 / 1.057 / 10.042 s','VQ path still uses block-wise H128']],64,112,832,178,[310,198,324],{size:20});
heading(s,'Native VQ speedup is not implemented',64,358,832,27,C.red);
text(s,'Profile transform / decoding costs and connect batched transforms.\nDevelop the native VQ path; measure end-to-end latency and memory.',64,408,832,62,{size:22});
note(s,'Different rollout trajectories yield different query counts; these means are not paired latency benchmarks.',479,15);

// 21/22 — generated design references retained separately from the editable figures.
for(const [title,file,caption] of [
 ['Appendix: ImageGen method design','method-imagegen-reference.png','Generated design reference; the method on slides 2–4 is rebuilt as editable PowerPoint objects.'],
 ['Appendix: ImageGen loss design','loss-imagegen-reference.png','Generated design reference; editable loss definitions and diagrams are on slides 11–13.']]){
 s=body(title,'本页是用户要求的built-inimagegen学术图视觉设计参考，图内label仅作示意，不作为数值证据。正式方法描述以slides2—4和11—13的可编辑objects与源码为准。'+src(ASSET+'/'+file,ASSET+'/imagegen-prompts.txt'));
 const b=await fs.readFile(ASSET+'/'+file);
 s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:title,fit:'contain',position:{left:48,top:78,width:864,height:392}});
 note(s,caption,480,14);
}

activeDiagram=null;
for(const o of original)o.delete();
for(let i=0;i<slides.length;i++)slides[i].moveTo(i);
if(slides.length!==22||arms.length!==11)throw new Error('Expected 22 slides and 11 completed Fast-WAM configurations');
await fs.mkdir(TMP+'/preview',{recursive:true});
await(await PresentationFile.exportPptx(p)).save(TMP+'/candidate.pptx');
await fs.writeFile(TMP+'/requirements.json',JSON.stringify({slides:slides.length,tableOwners,sourceHash:createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex')},null,2));
// Native draw.io source accompanies the editable PowerPoint reconstruction.
let drawio='<mxfile host="app.diagrams.net" version="24.7.17">';
for(const [name,cells]of diagrams){
 let xml='<mxGraphModel dx="960" dy="540" grid="0" page="1" pageScale="1" pageWidth="960" pageHeight="540"><root><mxCell id="0"/><mxCell id="1" parent="0"/>';
 let id=1;
 for(const c of cells){id++;
  if(c.type==='arrow')xml+=`<mxCell id="${id}" edge="1" parent="1" style="endArrow=${c.end?'block':'none'};strokeColor=${c.color};strokeWidth=${c.width};${c.dash?'dashed=1;':''}"><mxGeometry relative="1" as="geometry"><mxPoint x="${c.x1}" y="${c.y1}" as="sourcePoint"/><mxPoint x="${c.x2}" y="${c.y2}" as="targetPoint"/></mxGeometry></mxCell>`;
  else xml+=`<mxCell id="${id}" vertex="1" parent="1" value="${escape(c.text)}" style="rounded=${c.radius?1:0};whiteSpace=wrap;html=0;fillColor=${c.type==='box'?c.fill:'none'};strokeColor=${c.type==='box'?c.line:'none'};fontFamily=Arial;fontSize=${c.size};fontColor=${c.color};fontStyle=${c.bold?1:0};align=${c.align??'center'};verticalAlign=${c.type==='box'?'middle':'top'};"><mxGeometry x="${c.x}" y="${c.y}" width="${c.w}" height="${c.h}" as="geometry"/></mxCell>`;
 }
 drawio+=`<diagram name="${escape(name)}">${xml}</root></mxGraphModel></diagram>`;
}
drawio+='</mxfile>';await fs.writeFile(ASSET+'/editable-method-and-loss.drawio',drawio);
await fs.writeFile(ASSET+'/slide-outline.txt',slides.map((s,i)=>`${i+1}. ${s.shapes.items.find(x=>x.name==='Title 1')?.text??''}`).join('\n'));
console.log(JSON.stringify({candidate:TMP+'/candidate.pptx',slides:slides.length,tableOwners,diagrams:diagrams.size}));
