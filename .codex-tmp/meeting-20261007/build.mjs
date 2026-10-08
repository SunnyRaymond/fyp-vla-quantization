import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { importRuntimeModule } from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
import { finalizePresentation, makeNativeBulletParagraphs } from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
import { baseline, pipeline, geometry, ink, blue, teal, orange } from './diagrams.mjs';
process.on('uncaughtException',e=>{console.error(e.message);console.error((e.stack??'').split('\n').filter(x=>x.includes('build.mjs')).join('\n'));process.exit(1);});

const ROOT='D:/Downloads/Final Year Project';
const TMP=path.join(ROOT,'.codex-tmp/meeting-20261007');
const SOURCE='C:/Users/Raymond/.codex/skills/artifact-template-ntu-group-meeting/assets/reference.pptx';
const OUTPUT=path.join(ROOT,'Weekly Slides/2026-10-07_WAM_Quantization_Progress.pptx');
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const PYTHON='C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const { FileBlob, PresentationFile }=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const originals=[...p.slides.items];
const anchors=(await p.inspect({kind:'slide,textbox,shape,image',maxChars:100000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const coverSource=p.resolve(anchors.find(x=>x.kind==='slide'&&x.slide===1).id);
const bodySource=p.resolve(anchors.find(x=>x.kind==='slide'&&x.slide===6).id);
const local=JSON.parse(await fs.readFile(path.join(TMP,'results/slide_data.json'),'utf8'));
const paperAssets=JSON.parse(await fs.readFile(path.join(TMP,'papers/assets.json'),'utf8'));
const BLUE='#4C88DE', NAVY='#1B2059', GRAY='#697386', RED='#CB3141';
const slides=[];

function txt(s,name,text,x,y,w,h,{size=22,bold=false,color=ink,align='left',middle=false}={}){
 const sh=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=text; sh.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:align,verticalAlignment:middle?'middle':'top',autoFit:'none',wrap:'square',insets:{top:0,left:0,bottom:0,right:0}};
 return sh;
}
function body(title,notes){
 const s=bodySource.duplicate(), heading=s.shapes.items.find(x=>x.id==='2');
 for(const x of [...s.shapes.items])if(x!==heading&&x.id!=='3')x.delete();
 for(const x of [...s.images.items])x.delete();
 heading.text=title; heading.position={left:48,top:8,width:880,height:58};
 heading.text.style={typeface:'Arial',fontSize:112/3,bold:true,color:BLUE,alignment:'left',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
 s.speakerNotes.textFrame.setText(notes); slides.push(s); return s;
}
function foot(s,text){txt(s,'Evidence source',text,48,485,858,20,{size:12.5,color:GRAY});txt(s,'Slide number',String(slides.length).padStart(2,'0'),915,485,26,20,{size:12,color:GRAY});}
function bullet(s,name,items,x,y,w,h,size=22){const sh=txt(s,name,'',x,y,w,h,{size});sh.text=makeNativeBulletParagraphs(items,{marginLeftPoints:16,hangingPoints:8,spaceAfterPoints:10});return sh;}
function table(s,values,x,y,w,h,widths,{size=19,headerSize=size,total=false}={}){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:'#D4DCE6',style:'solid',width:0.8});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:size,color:ink},margins:{top:6,bottom:6,left:9,right:9},anchor:'center'});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:NAVY,textStyle:{typeface:'Arial',fontSize:headerSize,bold:true,color:'#FFFFFF'}});
 if(total)t.cells.block({row:values.length-1,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:'#EDF3FA',textStyle:{typeface:'Arial',fontSize:17.5,bold:true,color:ink}});
 return t;
}
async function picture(s,name,file,x,y,w,h,crop={left:0,top:0,right:0,bottom:0}){
 const bytes=await fs.readFile(path.join(TMP,'papers',file));
 const iw=bytes.readUInt32BE(16),ih=bytes.readUInt32BE(20),ew=iw*(1-crop.left-crop.right),eh=ih*(1-crop.top-crop.bottom);
 const k=Math.min(w/ew,h/eh),fr={left:x+(w-ew*k)/2,top:y+(h-eh*k)/2,width:ew*k,height:eh*k};
 s.images.add({blob:new Uint8Array(bytes),contentType:'image/png',alt:name,fit:'contain',position:fr,crop});
 return {frame:fr,scale:k,iw,ih};
}
function nativeDiagram(s,fig,x,y,k=1){
 const map=new Map();
 for(const n of fig.nodes){
  let sh;
  if(n.kind==='text')sh=txt(s,n.id,n.label,x+n.x*k,y+n.y*k,n.w*k,n.h*k,{size:(n.size??18)*k,bold:n.bold,color:n.color,middle:true});
  else{
   sh=s.shapes.add({name:n.id,geometry:n.kind==='ellipse'?'ellipse':'rect',position:{left:x+n.x*k,top:y+n.y*k,width:n.w*k,height:n.h*k},fill:n.fill,line:{fill:n.color,width:1.35*k}});
   sh.text=n.label;sh.text.style={typeface:'Arial',fontSize:n.size*k,color:n.color,bold:n.bold??false,alignment:'center',verticalAlignment:'middle',autoFit:'none',wrap:'square',insets:{top:4*k,bottom:4*k,left:4*k,right:4*k}};
  }
  map.set(n.id,sh);
 }
 for(const e of fig.edges)s.shapes.connect(map.get(e.from),map.get(e.to),{kind:e.kind??'straight',fromSide:e.fromSide??'right',toSide:e.toSide??'left',line:{fill:e.color??ink,width:(e.width??1.7)*k,style:e.dashed?'dashed':'solid'},tail:{type:'triangle',width:'sm',length:'sm'}});
}
const notesSource=f=>`本地来源：${path.join(ROOT,f).replaceAll('\\','/')}`;

const cover=coverSource.duplicate();for(const im of [...cover.images.items])im.delete();for(const sh of [...cover.shapes.items])sh.delete();
txt(cover,'Cover title','WAM Quantization',289.42,196,625,59,{size:44,bold:true,color:'#FFFFFF'});
txt(cover,'Cover subtitle','Progress & Next Steps',289.42,260,625,57,{size:39,bold:true,color:'#FFFFFF'});
txt(cover,'Meeting subtitle','Fast-WAM baseline → action-sensitive weight VQ',289.42,339,625,68,{size:23,bold:true,color:'#FFFFFF'});
txt(cover,'Meeting details','NTU · College of Computing and Data Science\nWeekly Group Meeting\n7 October 2026',289.42,420,625,80,{size:20,bold:true,color:'#FFFFFF'});
cover.speakerNotes.textFrame.setText('这次汇报分三部分：先确认 Fast-WAM Optional-IDM 的起点和两篇论文的 LIBERO 结果；再看我们自己的 LIBERO-Plus 子集与 A4 诊断；最后说明 action-sensitive weight VQ 的下一步方案。所有论文截图与本地数据分别标注来源。后面的 VQ 架构和 codebook 几何示例是 proposal，不是已完成结果。');slides.push(cover);

let s=body('Baseline: Fast-WAM Optional-IDM',`先用 Fast-WAM 的独立 ActionDiT 做可控、范围小的 weight VQ pilot。当前 Optional-IDM 实现分两段：先 denoise video，然后将完整 denoised video 固定为条件，prefill video K/V，再 denoise action。图按当前实现绘制，不把论文的 first-frame-only 快速路径混入本地 Optional-IDM。第一轮 VQ 只改 ActionDiT，video branch 保留 BF16，并不等同于前面的现有四臂联合 Linear 量化配置。\n之后推广到 LingBot-VA 和 Cosmos-Policy；共享 backbone 模型应称 action-targeted，不能假设仍是完全独立的 action-only 模块。\n来源：Fast-WAM arXiv:2603.16666v2；本地 FastWAM/src/fastwam/models/wan22/fastwam_idm.py，Stage 1/2 inference implementation。`);
nativeDiagram(s,baseline,55,105);foot(s,'Architecture follows the current Optional-IDM implementation; first VQ pilot targets ActionDiT.');

s=body('Fast-WAM on LIBERO: QuantWAMs',`这是 QuantWAMs Table 1 原始截图，Fast-WAM 所在 panel。LIBERO 平均 success：FP16 97.6±0.2%，SVDQuant W4A4 73.7±0.7%，SVDQuant* 75.3±0.6%，Atom 76.2±0.7%，Atom* 82.1±0.5%，QuantWAMs 97.4±0.2%。星号行按论文用于匹配 mixed-precision budget，不能当成与另一篇完全相同的运行 recipe。论文报告 LIBERO 每个 protocol seed 2,000 episodes，3 seeds 合计 6,000；截图中的 ± 是论文报告值。Speedup 与 memory 是该论文的实验口径，不是本地结果。\n来源：QuantWAMs: Calibrating at the Right Granularity for World Action Models, arXiv:2607.28405v1, Table 1, PDF physical p.7。https://arxiv.org/abs/2607.28405v1`);
txt(s,'Paper evidence label','Paper-reported result · Table 1',55,86,845,31,{size:20,color:GRAY});
await picture(s,'QuantWAMs Table 1: Fast-WAM RoboTwin and LIBERO','quantwams-table1-fast-wam.png',35,137,890,225);
txt(s,'SVDQuant reference','SVDQuant W4A4: 73.7 ± 0.7% LIBERO Avg',55,394,850,37,{size:27,bold:true,color:orange});
txt(s,'QuantWAM result','QuantWAMs W4A4: 97.4 ± 0.2%',55,438,850,35,{size:24,color:ink});
foot(s,'QuantWAMs · arXiv:2607.28405v1 · Table 1, p.7 · Original PDF pixels');

s=body('Fast-WAM on LIBERO: SteerQuant',`这是 SteerQuant Table 1 的 FastWAMJoint 部分，保留原 PDF pixels。为了可读性，zoom 从同一页拼接左侧 Precision/Method 与右侧 FastWAMJoint panel，聚焦 header 与 W4A4 rows，没有改写数字。红圈标出 SVDQuant W4A4 的 Avg 98.2%。上一页 QuantWAMs 同名 baseline 是 73.7±0.7%，mean 相差 24.5 个百分点；Atom 也存在较大差异。现阶段把它作为 cross-paper protocol alignment 问题，不能仅凭两个表宣称论文或数据有误。需要先对齐模型变体 / checkpoint、量化范围、calibration、mixed precision 和 evaluation seeds。SteerQuant caption 报告 2,000 LIBERO rollouts。\n来源：SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models, arXiv:2609.39056v1, Table 1, PDF physical p.7。https://arxiv.org/abs/2609.39056v1`);
txt(s,'Paper evidence label','Paper-reported result · FastWAMJoint W4A4 rows, Table 1',55,84,850,31,{size:20,color:GRAY});
const zoomAsset=paperAssets.assets.find(a=>a.file==='steerquant-table1-fastwamjoint-panel.png');
const zoomFile='steerquant-table1-w4a4-zoom.png';
const z=await picture(s,'SteerQuant original Table 1 pixels: Method plus FastWAMJoint panel',zoomFile,55,128,850,284);
const oldBox=zoomAsset.target_cell.glyph_bbox_px_in_this_image;
const box={...oldBox,y:oldBox.y-438+151};
if(!box)throw new Error('Missing SteerQuant target-cell pixel bbox');
s.shapes.add({name:'Circle SVDQuant W4A4 Avg 98.2',geometry:'ellipse',position:{left:z.frame.left+(box.x-5)*z.scale,top:z.frame.top+(box.y-4)*z.scale,width:(box.width+10)*z.scale,height:(box.height+8)*z.scale},fill:'none',line:{fill:RED,width:2.8}});
txt(s,'Cross-paper discrepancy','SVDQuant W4A4: 98.2% here vs 73.7% in QuantWAMs',55,430,850,33,{size:25,bold:true,color:RED});
txt(s,'Alignment question','Check model variant, quantization scope and evaluation protocol.',55,465,850,24,{size:18,color:GRAY});
foot(s,'SteerQuant · arXiv:2609.39056v1 · Table 1, p.7 · Same-page pixel zoom');

s=body('Our LIBERO-Plus pilot: observed subset',local.slides[0].speaker_notes_zh+'\n现有四臂使用相同 W4 G128 weight bank 与 dynamic per-row RTN A8/A4，量化614个Linear（video/action/proprio），不是前页拟议的 ActionDiT-only VQ。Text/VAE、norm、非Linear和attention运算保持BF16。没有完整benchmark结论，也没有正式部署latency。\n'+notesSource(local.plus.source_file));
txt(s,'Pilot subtitle','300 paired variants · Spatial only · 6 of 28 planned cells',55,82,850,36,{size:22,bold:true});
const labels={background_textures:'Background textures',camera_viewpoints:'Camera viewpoints',language_instructions:'Language instructions',light_conditions:'Light conditions',objects_layout:'Objects layout',robot_initial_states:'Robot initial states'};
const fmt=(v,n)=>`${v}/${n} (${Math.round(100*v/n)}%)`;
const vals=[['Spatial dimension','BF16','W4A8','W4A4','W4A4KV4'],...local.plus.observed_cells_table.map(r=>[`${labels[r.suite_dimension.split(' × ')[1]]} (n=50)`,...['BF16','W4A8','W4A4','W4A4KV4'].map(a=>fmt(r[a],r.n))]),['Overall (n=300)',...local.plus.observed.arm_success_summary.map(r=>fmt(r.successes,r.episodes))]];
table(s,vals,55,135,850,292,[310,142,142,128,128],{size:18.5,headerSize:18.5,total:true});
txt(s,'Pilot observation','W4A8: +3 pp descriptively; both A4 arms: 0/300.',55,443,850,30,{size:23,bold:true});
foot(s,'Local snapshot: 6 Oct 2026 · All completed shards · Joint Linear quantization · 22 cells have no results');

s=body('A4 diagnosis: error appears before rollout',local.slides[1].speaker_notes_zh+'\n'+notesSource(local.plus.source_file)+'\n'+notesSource(local.diagnose_a4.source_file));
txt(s,'Reset panel title','300 reset-observation pairs',55,86,406,30,{size:23,bold:true,color:blue});
txt(s,'Reset panel definition','Median normalized motor RMSE · first 10 actions',55,122,406,48,{size:17,color:GRAY});
table(s,[['Comparison','RMSE'],['W4A8 vs BF16','0.0228'],['W4A4 vs BF16','0.3222'],['W4A4KV4 vs BF16','0.3217'],['KV4 addition vs W4A4','0.0387']],55,181,406,225,[281,125],{size:18});
txt(s,'Native panel title','One-input native-path check',495,86,410,30,{size:23,bold:true,color:teal});
txt(s,'Native panel definition','Same input · 4 queries · no rollout steps',495,122,410,48,{size:17,color:GRAY});
table(s,[['Diagnostic measure','Value'],['W4A4 vs BF16 motor RMSE¹','0.2853'],['Native vs FP32 ref. RMSE²','0.0407'],['INT4 GEMM calls','6,436'],['Discrete gripper flips¹','0/10']],495,181,410,225,[297,113],{size:18});
txt(s,'Diagnosis interpretation','The A4 recipe needs diagnosis; current controls do not isolate W-only or A-only effects.',55,430,850,47,{size:21,bold:true});
foot(s,'¹ First 10 action positions. ² Full 32×7 chunk, all channels. Fixed-input fidelity; no timing claim.');

s=body('QuantWAMs: calibrate at the right granularity',`主要 figure 是 QuantWAMs Figure 2 原图。可按左中右说明：shared-basis calibration 只在 coordinate-compatible modules 合并activation statistics，不能随意把Q/K/V跨private streams合并；co-training-guided saliency 用joint video-action objective排序层，配置mixed precision；real-rollout counterfactual audit 用固定干预对denoising-step protection做closed-loop验证。启发是：量化误差的重要性要按系统结构与控制结果判断，不能只用activation amplitude。\n来源：QuantWAMs, arXiv:2607.28405v1, Figure 2。https://arxiv.org/abs/2607.28405v1`);
await picture(s,'QuantWAMs Figure 2 main architecture','quantwams-fig2-method-main.png',40,85,880,370);
txt(s,'Paper takeaway','Structure-aware calibration + joint saliency + rollout validation',55,457,850,27,{size:21,bold:true});foot(s,'QuantWAMs · arXiv:2607.28405v1 · Figure 2 · Figure crop from original PDF');

s=body('Q-WAM: protect action-sensitive subspaces',`Q-WAM Figure 2 原图。AOG（Action Observability Gramian）估计layer输入误差经过后续denoising传播到最终action的敏感方向；ASP把最敏感子空间留在16-bit branch，剩余部分4-bit。图中的95% action mass/17% action-expert parameters是论文对应设置下的统计，不是所有WAM通用常数。其benchmark主要是RoboTwin与real-world，不把这篇列成Fast-WAM LIBERO表的来源。\n对于我们的VQ方案，action sensitivity已有先例，不能把“关注action敏感方向”单独视作novelty。下一步要验证敏感度是否能指导weight codebook，而不是机械复制一个高精度残差branch。\n来源：Q-WAM: 4-Bit Quantization of World Action Models with Action-Subspace Protection, arXiv:2609.33269v1, Figure 2。https://arxiv.org/abs/2609.33269v1`);
await picture(s,'Q-WAM Figure 2 AOG and ASP','q-wam-fig2-method-main.png',40,82,880,379);
txt(s,'Paper takeaway','Measure action sensitivity; preserve the most influential directions.',55,457,850,27,{size:21,bold:true});foot(s,'Q-WAM · arXiv:2609.33269v1 · Figure 2 · Figure crop from original PDF');

s=body('SteerQuant: route error away from actions',`SteerQuant Figure 2 原图。先估计layer × denoising timestep × stream的action-impact map，再用shared-weight scaling与stream-specific activation modulation把量化误差转移到较低action-impact的stream。Online inference 的 Rudder 把 scaling、量化/反量化或output compensation与low-bit GEMM融合。它不是已实现的VQ lookup kernel，后续我们若做VQ需要设计codebook/index表示和lookup/decode/GEMM的路径。\n来源：SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models, arXiv:2609.39056v1, Figure 2。https://arxiv.org/abs/2609.39056v1`);
await picture(s,'SteerQuant Figure 2 error routing and Rudder','steerquant-fig2-method-main.png',40,81,880,380);
txt(s,'Paper takeaway','Action-impact calibration → error routing → fused low-bit inference',55,457,850,27,{size:21,bold:true});foot(s,'SteerQuant · arXiv:2609.39056v1 · Figure 2 · Figure crop from original PDF');

s=body('Next step: action-sensitive weight VQ',`这是待验证方案，不是实测结论。第一阶段先做weight VQ，activation维持BF16，使codebook效果可单独评估；之后再试A8/A4。SmoothQuant式D缩放与Hadamard H作用于同一个输入feature维度：若Y=XWᵀ，设X̃=XD⁻¹H、W̃=WDH，并满足D可逆、HHᵀ=I，则量化前X̃W̃ᵀ=XWᵀ。这个配对变换把缩放的负担转到weight branch，再用VQ处理变换后的weight groups。Hadamard只保证正交与norm保持，不能直接保证每个operator的outlier下降；需测activation tails、weight tails和operator placement。\nAction sensitivity M从calibration估计，用于偏向保护敏感方向的codebook与assignment。必须和uniform/L2 codebook在相同bit/byte预算下比较，并记录closed-loop success，避免仅看weight reconstruction MSE。图中decode是概念路径，第一轮可先做fidelity验证；performance需要packed表示与native kernel。\n来源：SmoothQuant https://proceedings.mlr.press/v202/xiao23c.html ; QuaRot https://papers.nips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html ; VPTQ https://aclanthology.org/2024.emnlp-main.467/ 。`);
nativeDiagram(s,pipeline,55,85);txt(s,'Equivalent identity','Before quantization:   X̃W̃ᵀ = XWᵀ   when HHᵀ = I and D is invertible.',55,440,850,31,{size:21,bold:true});foot(s,'Proposed method · BF16 activations first; A8/A4 later · Outlier reduction must be measured');

s=body('Action-sensitive codebook geometry',`这是人工构造的二维示意图，不是实验点或论文结果。权重block w=(0,0)，两个候选codewords：c₁=(0.3,0)、c₂=(0,0.6)。普通Euclidean distortion分别为0.09与0.36，会选择c₁。设action sensitivity metric M=diag(9,1)，action-weighted distortion分别为0.81与0.36，选择c₂。尽管c₂的Euclidean距离更远，它沿更不敏感方向引入误差。椭圆表示固定action-weighted error水平集；敏感方向容许更小偏差。\n实际codebook需优化Σⱼ(w̃ⱼ−c_zj)ᵀMⱼ(w̃ⱼ−c_zj)。可用M≈E[JᵀJ]（J从变换后weight group到最终action的Jacobian，包含denoising），或低秩/梯度PSD proxy。逐组求和忽略cross-group terms，是local approximation。所有metric与codebook必须在同一transformed coordinates中定义。\n第一轮消融：L2 VQ → +D/H → +action metric，固定storage预算与paired protocol。只有weight误差小不保证控制更好；需看action fidelity和paired closed-loop outcomes。`);
nativeDiagram(s,geometry,55,87);txt(s,'Codebook objective','L = Σⱼ (w̃ⱼ − C[zⱼ])ᵀ Mⱼ (w̃ⱼ − C[zⱼ])',55,441,850,36,{size:25,bold:true});foot(s,'Illustrative geometry, not measured data · M is an estimated PSD action-sensitivity metric');

s=body('After VQ: cache, kernels and QAT',`先完成weight VQ的fidelity与closed-loop gate，再扩展：\n1. KV cache quantization：把K/V视为persistent activation，量化对象与weight不同。先分开测试，避免沿用A4已经崩溃时的组合臂来归因KV4。验证cache error、packed bytes与控制结果。\n2. Rudder-inspired VQ kernel：学习其融合思路，但重新设计codebook/index packed representation、lookup/decode、GEMM、scaling compensation以及模型集成。fake quant或离线压缩不等于native speedup；正式结论需paired native end-to-end latency与relevant fidelity。\n3. QAT：resource-dependent extension。已核对的QuantWAMs、Q-WAM、SteerQuant都是PTQ。当前不宣称“全球没有人做WAM QAT”或已经证明novelty。若有GPU且PTQ诊断支持，先在Fast-WAM独立ActionDiT做小范围quantization-aware codebook/weight tuning，再推广到共享模型。需要训练预算和可复现的held-out control evaluation。\n顺序：先数学变换和error mechanism → bounded fidelity pilot → paired closed-loop → packed/native performance。`);
txt(s,'Cache heading','1   KV cache quantization',64,96,830,36,{size:28,bold:true,color:blue});
txt(s,'Cache body','Test K/V precision after weight VQ; measure cache bytes and control fidelity.',100,145,790,54,{size:23});
txt(s,'Kernel heading','2   Rudder-inspired VQ kernels',64,227,830,36,{size:28,bold:true,color:teal});
txt(s,'Kernel body','Fuse codebook lookup / decode with GEMM and scaling; test native end-to-end latency.',100,276,790,66,{size:23});
txt(s,'QAT heading','3   QAT, if GPU budget permits',64,363,830,36,{size:28,bold:true,color:orange});
txt(s,'QAT body','Start with an ActionDiT pilot; direct WAM-control QAT remains a direction to investigate.',100,412,790,66,{size:23});
foot(s,'Order: weight-VQ fidelity → paired closed-loop evaluation → packed kernels → resource-dependent QAT');

for(const original of originals)original.delete();for(let i=0;i<slides.length;i++)slides[i].moveTo(i);
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});
const candidate=path.join(TMP,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const refHash=createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:OUTPUT,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-bullet-geometry','--validate-heading-fit',...[5,6].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[5,6],fontPolicy:{basis:'reference',families:['Arial'],referencePath:SOURCE,referenceSha256:refHash},verifyArtifactToolImport:true,receiptPath:path.join(TMP,'validation-final.json')});
console.log(JSON.stringify({slideCount:slides.length,output:OUTPUT,status:result.status??'complete'}));
const final=await PresentationFile.importPptx(await FileBlob.load(OUTPUT));await fs.mkdir(path.join(TMP,'final-preview'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++){const b=await final.export({slide:final.slides.items[i],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'final-preview',`slide-${i+1}.png`),Buffer.from(await b.arrayBuffer()));}
// Preserve reusable, editable diagram sources with the presentation.
for(const file of ['fast-wam-idm-baseline.drawio','weight-vq-pipeline.drawio','action-sensitive-codebook.drawio'])await fs.copyFile(path.join(TMP,'method-figures',file),path.join(ROOT,'Weekly Slides',`2026-10-07_${file}`));


