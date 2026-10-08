import fs from 'node:fs/promises';
import path from 'node:path';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
import {finalizePresentation} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
import {methodA,geometryA,methodB,geometryB} from './figures.mjs';
process.on('uncaughtException',e=>{console.error(e.message);console.error((e.stack??'').split('\n').filter(x=>x.includes('revise.mjs')||x.includes('figures.mjs')).join('\n'));process.exit(1);});
const ROOT='D:/Downloads/Final Year Project',TMP=path.join(ROOT,'.codex-tmp/meeting-20261007-revision');
const SOURCE=path.join(ROOT,'Weekly Slides/2026-10-07_WAM_Quantization_Progress.pptx');
const OUTPUT=path.join(ROOT,'Weekly Slides/2026-10-07_WAM_Quantization_Progress_Revised.pptx');
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const PYTHON='C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(TMP,'source-no-old-tables.pptx')));
const originalSlides=[...p.slides.items];
const anchors=(await p.inspect({kind:'slide,shape,textbox,table',maxChars:180000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
const slide=n=>p.resolve(anchors.find(x=>x.kind==='slide'&&x.slide===n).id);
const BLUE='#4C88DE',NAVY='#1B2059',GRAY='#697386';
function txt(s,name,text,x,y,w,h,{size=22,bold=false,color='#233044'}={}){const a=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});a.text=text;a.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{top:0,left:0,bottom:0,right:0}};return a;}
function setText(s,name,text,size){const a=s.shapes.items.find(x=>x.name===name);if(!a)throw new Error('Missing source shape '+name);a.text=text;if(size)a.text.style={typeface:'Arial',fontSize:size,color:GRAY,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};return a;}
function table(s,values,x,y,w,h,widths,{size=17,header=16}={}){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.borders.assign({fill:'#D4DCE6',width:.7,style:'solid'});t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:size,color:'#233044'},margins:{top:5,bottom:5,left:8,right:8},anchor:'center'});t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:NAVY,textStyle:{typeface:'Arial',fontSize:header,bold:true,color:'#FFFFFF'}});return t;}

// Replace only the right diagnostic panel. Keep the paired-observation evidence.
const d=slide(6);
setText(d,'Reset panel title','300 reset-observation pairs').position={left:55,top:86,width:330,height:30};
const leftDef=setText(d,'Reset panel definition','Median normalized motor RMSE\nFirst 10 action positions',17);leftDef.position={left:55,top:122,width:330,height:47};
setText(d,'Native panel title','Activation encoding: A4 vs A8').position={left:404,top:86,width:501,height:30};
const rightDef=setText(d,'Native panel definition','Zero-code ratio and input relative RMSE',18);rightDef.position={left:404,top:122,width:501,height:35};
table(d,[['Comparison','RMSE'],['W4A8 vs BF16','0.0228'],['W4A4 vs BF16','0.3222'],['W4A4KV4 vs BF16','0.3217'],['KV4 addition vs W4A4','0.0387']],55,180,330,230,[242,88],{size:18,header:18});
const encoding=[
 ['Call region','A4 zero\ncodes','A8 zero\ncodes','A4 rel.\nRMSE','A8 rel.\nRMSE'],
 ['Video prefill','71.52%','8.52%','41.54%','2.97%'],
 ['Video step 0','69.37%','8.45%','38.69%','2.88%'],
 ['Video step 9','65.39%','7.97%','37.39%','2.79%'],
 ['Action step 0','55.98%','6.07%','31.49%','2.03%'],
 ['Action step 9','58.77%','6.32%','31.49%','1.97%'],
 ['Proprio','13.64%','0.57%','6.50%','0.33%']
];
table(d,encoding,404,167,501,269,[153,87,87,87,87],{size:16.5,header:15.5});
const takeaway=setText(d,'Diagnosis interpretation','High A4 zero-code ratios accompany larger input RMSE in these calls.');takeaway.position={left:55,top:452,width:850,height:30};takeaway.text.style={typeface:'Arial',fontSize:21,bold:true,color:'#233044',autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};
const evidence=setText(d,'Evidence source','Right: user-provided activation diagnostic. Step 0/9 denotes denoising iterations.',12.5);evidence.position={left:48,top:485,width:858,height:20};
d.speakerNotes.textFrame.setText('左侧保留原 n=300 paired reset-observation action-output evidence，motor RMSE 是前10个action位置的 median normalized RMSE。右侧按照用户提供的截图逐值转录，表中量化调用区域分别为 Video conditioning prefill、Video denoising step 0/9、Action denoising step 0/9、Proprio conditioning。所有百分比与截图保持一致；缩短行名只为版面，step 0/9 是 diffusion denoising iterations，与左侧动作块前10位置不同。零编码比例是量化为zero code的比例，不能直接等同于整个输入张量都为零。输入relative RMSE是截图中的activation/input指标，不能与左侧action-output RMSE混为一个指标或假定同一分母。截图没有给样本数、聚合范围或reference细节，因此不补造这些定义，也不将右侧标成 n=300。观察是这些调用中A4的zero-code ratio与input error都明显大于A8，支持进一步检验activation quantizer的动态范围和尺度分配。\n右表来源：用户截图 D:/Downloads/Final Year Project/Weekly Slides/2026-10-07_Figure_Designs/activation-diagnostic-reference.png。左表来源：experiment/idea-validation/fastwam-libero-plus-pilot/artifacts/25699421.pbs101/results/analysis.json#/supplementary/first_action。');

const originalFuture=slide(12),src=slide(10),added=[];
function newPage(title,notes,figure,foot){const s=src.duplicate();for(const q of [...s.shapes.items])if(q.name!=='Title 1'&&q.name!=='矩形 2')q.delete();for(const i of [...s.images.items])i.delete();for(const t of [...s.tables.items])t.delete();const titleShape=s.shapes.items.find(q=>q.name==='Title 1');titleShape.text=title;titleShape.position={left:48,top:8,width:880,height:58};titleShape.text.style={typeface:'Arial',fontSize:112/3,bold:true,color:BLUE,autoFit:'none',insets:{top:0,bottom:0,left:0,right:0}};figure(s);txt(s,'Evidence source',foot,48,485,858,20,{size:12.5,color:GRAY});txt(s,'Slide number','',915,485,26,20,{size:12,color:GRAY});s.speakerNotes.textFrame.setText(notes);added.push(s);return s;}
const methodNotes='这是 weight VQ proposal，所有矩阵色块、codewords与indices仅为概念示意，不是校准统计或实测结果。行向量约定 Y=XWᵀ，在同一个输入feature维上取可逆对角D和正交H，X′=XD⁻¹H、W′=WDH，则量化前X′(W′)ᵀ=XWᵀ。D采用SmoothQuant-style scaling，Hadamard旋转H只保证正交和norm保持，activation outlier/tail是否下降需要测量。先做BF16-activation的weight VQ pilot，再按实验支持扩展A8/A4。每个transformed output row分成若干weight vectors，codebook C和离散indices z用action-sensitive distortion拟合和分配。M应由从对应transformed weight group到最终action的Jacobian/PSD proxy估计，可以包含denoising propagation，不能误当成普通activation-only MSE。逐group的目标忽略cross-group terms，是局部近似。可写 M≈E[JᵀJ]，dₘ(w′,c)=(w′−c)ᵀM(w′−c)。图中的VQ linear仍位于ActionDiT内部，其输出经remaining layers/denoising才成为最终action chunk。离线重建/压缩与native lookup/GEMM speedup是不同验证阶段，图不宣称已实现VQ speedup。\n参考：SmoothQuant https://proceedings.mlr.press/v202/xiao23c.html ; QuaRot https://papers.nips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html ; VPTQ https://aclanthology.org/2024.emnlp-main.467/ 。';
const geometryNotes='二维示意例子：w=(0,0)，c₁=(0.3,0)，c₂=(0,0.6)，普通Euclidean squared distances为0.09、0.36，所以选c₁。M=diag(9,1)下，action-weighted distances为0.81、0.36，所以选c₂。水平坐标更敏感，垂直坐标更不敏感；equal-action-error ellipse垂直/水平radius比为3。ellipse在distortion=0.36时水平半径0.2、垂直半径0.6；c₂位于边界，c₁位于椭圆外。L2 nearest-neighbor circle取半径0.3，c₁位于边界，c₂在外。图内的点、contours、bars和codebook spacing仅是这个数学示例，不是实验数据或已验证性能。实际codebook的分布需要校准数据优化，不能仅凭示意网格推断真实收益。';
newPage('Weight VQ: proposal A',methodNotes+'\n图形来源：本研究 proposed method illustration。',methodA,'Proposed method. Matrices and assignments are schematic. BF16 activations for the first weight-VQ pilot.');
newPage('Codebook geometry: proposal A',geometryNotes+'\n图中contour plot、距离条形示例和metric-codebook示意使用同一组数学定义。',geometryA,'Illustrative example. The action metric penalizes errors more strongly along the horizontal direction.');
newPage('Weight VQ: proposal B',methodNotes+'\nVQ linear的中间输出需要经过remaining ActionDiT / denoising得到最终action。设计参考图保存在Weekly Slides/2026-10-07_Figure_Designs/imagegen-method-reference.png。',methodB,'Proposed method. An action-sensitive metric guides vector assignments. Packed-kernel performance remains to test.');
newPage('Codebook geometry: proposal B',geometryNotes+'\n两个plot采用相同坐标与candidates，按各自metric显示选择结果。设计参考图保存在Weekly Slides/2026-10-07_Figure_Designs/imagegen-geometry-reference.png。',geometryB,'Illustrative example. Identical candidates receive different assignments under L₂ and action-weighted distortion.');
const desiredOrder=[...originalSlides.slice(0,11),...added,originalFuture];
for(const s of [...desiredOrder].reverse())s.moveTo(0);
for(let i=0;i<p.slides.items.length;i++){const s=p.slides.items[i];const num=s.shapes.items.find(x=>x.name==='Slide number');if(num)num.text=String(i+1).padStart(2,'0');}
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});
const candidate=path.join(TMP,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
// Render the candidate before publishing so any visual repairs stay private.
await fs.mkdir(path.join(TMP,'preview'),{recursive:true});
const verify=await PresentationFile.importPptx(await FileBlob.load(candidate));
for(const n of [6,12,13,14,15]){const png=await verify.export({slide:verify.slides.items[n-1],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'preview',`slide-${n}.png`),Buffer.from(await png.arrayBuffer()));}
if(process.argv.includes('--draft-only')){console.log(JSON.stringify({candidate,slides:p.slides.items.length}));process.exit(0);}
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:OUTPUT,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit','--require-native-table-slide','5','--require-native-table-slide','6'],requiredNativeTableOwnerSlides:[5,6],fontPolicy:{basis:'design',families:['Arial','Times New Roman']},verifyArtifactToolImport:true,receiptPath:path.join(TMP,'final.validation.json')});
console.log(JSON.stringify({output:OUTPUT,slides:p.slides.items.length,status:result.status??'complete'}));
const final=await PresentationFile.importPptx(await FileBlob.load(OUTPUT));await fs.mkdir(path.join(TMP,'final-preview'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++){const png=await final.export({slide:final.slides.items[i],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'final-preview',`slide-${i+1}.png`),Buffer.from(await png.arrayBuffer()));}

