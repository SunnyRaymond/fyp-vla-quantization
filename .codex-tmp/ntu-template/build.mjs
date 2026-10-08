import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { importRuntimeModule } from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
import { finalizePresentation, makeNativeBulletParagraphs } from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';

const ROOT='D:/Downloads/Final Year Project';
const TMP=path.join(ROOT,'.codex-tmp/ntu-template');
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const SOURCE='D:/Downloads/Presentation_GuoYichen.pptx';
const OUTPUT=path.join(ROOT,'Weekly Slides/NTU_Group_Meeting_Template.pptx');
const PYTHON='C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const { FileBlob, PresentationFile }=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const originals=[...p.slides.items];
const sourceAnchors=JSON.parse(JSON.stringify((await p.inspect({kind:'slide,textbox,shape,image',maxChars:100000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x))));
const coverSource=p.resolve(sourceAnchors.find(x=>x.kind==='slide'&&x.slide===1).id);
const bodySource=p.resolve(sourceAnchors.find(x=>x.kind==='slide'&&x.slide===6).id);
const closingSource=p.resolve(sourceAnchors.find(x=>x.kind==='slide'&&x.slide===13).id);
const BLUE='#4C88DE', NAVY='#1B2059', GRAY='#697386';
function text(slide,name,content,x,y,w,h,{size=24,bold=false,color='#111111',center=false,placeholder=false}={}){
 const s=slide.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0},...(placeholder?{placeholderType:'body'}:{})});
 s.text=content;
 s.text.style={typeface:'Arial',fontSize:size,bold,color,alignment:center?'center':'left',verticalAlignment:'top',autoFit:'none',wrap:'square',insets:{top:0,bottom:0,left:0,right:0}};
 return s;
}
function bullets(slide,name,items,x,y,w,h,size=24){
 const s=text(slide,name,'',x,y,w,h,{size,placeholder:true});
 s.text=makeNativeBulletParagraphs(items,{marginLeftPoints:18,hangingPoints:9,spaceAfterPoints:12});
 return s;
}
function body(title,notes){
 const s=bodySource.duplicate();
 const heading=s.shapes.items.find(x=>x.id==='2');
 for(const x of [...s.shapes.items]) if(x!==heading&&x.id!=='3') x.delete();
 for(const x of [...s.images.items]) x.delete();
 heading.text=title;
 heading.position={left:48,top:8,width:880,height:56};
 heading.text.style={typeface:'Arial',fontSize:112/3,bold:true,color:BLUE,alignment:'left',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
 s.speakerNotes.textFrame.setText(notes);
 return s;
}
function figure(slide,label,x,y,w,h){
 const s=text(slide,label,`[${label}]`,x,y,w,h,{size:24,color:GRAY,center:true,placeholder:true});
 s.text.verticalAlignment='middle';
 s.line={style:'dashed',fill:'#B7C1CE',width:1};
 return s;
}
function table(slide,values,x,y,w,h,widths){
 const t=slide.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:'#CBD3DF',style:'solid',width:1});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:'#FFFFFF',textStyle:{typeface:'Arial',fontSize:22,color:'#111111'},margins:{top:10,bottom:10,left:14,right:14},anchor:'center'});
 t.cells.block({row:0,column:0,rowCount:1,columnCount:values[0].length}).assign({fill:NAVY,textStyle:{typeface:'Arial',fontSize:22,bold:true,color:'#FFFFFF'}});
 return t;
}
const slides=[];
const cover=coverSource.duplicate();
for(const i of [...cover.images.items])i.delete();
const ct=cover.shapes.items.find(x=>x.id==='2');
ct.text='[Research topic]';
ct.position={left:289.42,top:196,width:625,height:102};
ct.text.style={typeface:'Arial',fontSize:128/3,bold:true,color:'#FFFFFF',alignment:'left',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
text(cover,'Meeting subtitle','Weekly Group Meeting',289.42,306,625,40,{size:80/3,bold:true,color:'#FFFFFF'});
const ca=cover.shapes.items.find(x=>x.id==='3');
ca.text='[Presenter name]\nCollege of Computing and Data Science (CCDS)\n[Meeting date]';
ca.position={left:289.42,top:394.57,width:625,height:90};
ca.text.style={typeface:'Arial',fontSize:20,bold:true,color:'#FFFFFF',alignment:'left',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
cover.speakerNotes.textFrame.setText('填写 Research topic、Presenter name 和 Meeting date。长标题可使用两行。');
slides.push(cover);

let s=body('Meeting Outline','按实际汇报内容修改或删除条目。每次组会可选择需要的页面。');
bullets(s,'Agenda',['Research progress','Method and experimental setup','Results and interpretation','Open questions and next steps'],64,108,828,340,28);slides.push(s);

s=body('Weekly Progress','用简短条目区分已完成工作、当前进展与待解决问题。');
text(s,'Completed heading','Completed',64,100,810,36,{bold:true});
bullets(s,'Completed work',['[Completed task and outcome]','[Completed task and outcome]'],64,147,825,100);
text(s,'In progress heading','In progress',64,274,810,36,{bold:true});
bullets(s,'Current work',['[Current task and status]'],64,322,825,60);
text(s,'Open question heading','Open question',64,401,810,36,{bold:true});
text(s,'Open question','[Issue that needs discussion]',84,445,800,40);slides.push(s);

s=body('Research Question','填写本次研究问题、已有方法的限制和可检验的假设。');
text(s,'Question heading','Question',64,100,800,34,{bold:true});
text(s,'Research question','[Specific question for this study]',64,147,825,58);
text(s,'Motivation heading','Motivation',64,235,800,34,{bold:true});
bullets(s,'Motivation',['[Limitation of the current method]','[Why this limitation matters]'],64,282,825,100);
text(s,'Hypothesis heading','Hypothesis',64,410,800,34,{bold:true});
text(s,'Hypothesis','[Testable expectation]',64,454,825,40);slides.push(s);

s=closingSource.duplicate();
for(const i of [...s.images.items])i.delete();
let sh=s.shapes.items.find(x=>x.id==='2');sh.text='[Section title]';sh.position={left:290,top:228,width:620,height:94};sh.text.style={typeface:'Arial',fontSize:40,bold:true,color:'#FFFFFF',alignment:'center',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
text(s,'Section subtitle','[Short section description]',290,340,620,60,{size:24,color:'#FFFFFF',center:true});
s.speakerNotes.textFrame.setText('可选章节分隔页，可复制到需要的位置或删除。');slides.push(s);

s=body('Method Overview','将图位替换为实际 method figure。图中 labels 和公式应保证可读，caption 写清楚图的来源或定义。');
text(s,'Method heading','Approach',64,105,335,34,{bold:true});
bullets(s,'Method explanation',['[Core idea]','[Change from the baseline]','[Expected effect]'],64,155,335,255);
figure(s,'Insert method figure',435,101,460,314);
text(s,'Method caption','[Figure caption and source]',435,433,460,44,{size:16,color:GRAY});slides.push(s);

s=body('Experimental Setup','填写完整实验条件。至少说明 benchmark、comparison、evaluation protocol、runs / seeds 和 hardware。');
table(s,[['Setting','Specification'],['Task / benchmark','[Task and dataset]'],['Method / baseline','[Methods being compared]'],['Evaluation protocol','[Split and evaluation conditions]'],['Runs / seeds','[Number of runs and seeds]'],['Hardware','[Device and relevant configuration]']],64,105,832,345,[260,572]);
text(s,'Setup note','[Fixed conditions and measurement scope]',64,468,832,34,{size:18,color:GRAY});slides.push(s);

s=body('Results Table','将占位值替换为实际结果。Metric 标题注明单位和更好方向；caption 写明样本数、uncertainty 和测量范围。不要把示例占位符当作数据。');
table(s,[['Method','[Metric 1]','[Metric 2]','[Cost / latency]'],['[Baseline]','[Value]','[Value]','[Value]'],['[Variant A]','[Value]','[Value]','[Value]'],['[Variant B]','[Value]','[Value]','[Value]']],64,110,832,242,[280,184,184,184]);
text(s,'Result interpretation heading','Observation',64,385,820,34,{bold:true});
text(s,'Result interpretation','[Main observation supported by the table]',64,431,832,50);
text(s,'Result scope','[Number of runs, uncertainty and measurement scope]',64,487,832,20,{size:14,color:GRAY});slides.push(s);

s=body('Results Figure','替换为真实 plot、qualitative result 或 rollout frames。在左侧给出结果解释与适用范围。');
text(s,'Figure takeaway heading','Observation',64,105,320,34,{bold:true});
bullets(s,'Figure takeaway',['[Main observation]','[Comparison or trend]','[Limit of this conclusion]'],64,153,320,290);
figure(s,'Insert results figure',421,104,475,327);
text(s,'Result figure caption','[Caption, units and source]',421,449,475,44,{size:16,color:GRAY});slides.push(s);

s=body('Limitations and Open Questions','说明当前证据的边界与下一步需要验证的事项。可将具体问题留给组会讨论。');
text(s,'Limitations heading','Limitations',64,100,820,34,{bold:true});
bullets(s,'Limitations',['[Condition not yet tested]','[Uncertainty that affects interpretation]'],64,148,832,160);
text(s,'Discussion heading','Questions for discussion',64,340,820,34,{bold:true});
bullets(s,'Discussion questions',['[Decision or feedback needed]','[Alternative worth considering]'],64,389,832,105);slides.push(s);

s=body('Next Steps','填写下一阶段的工作及完成判据。可根据需要添加具体日期。');
table(s,[['Task','Expected outcome','Target date'],['[Task 1]','[Completion criterion]','[Date]'],['[Task 2]','[Completion criterion]','[Date]'],['[Task 3]','[Completion criterion]','[Date]']],64,109,832,278,[240,424,168]);
text(s,'Dependency heading','Dependency',64,424,820,34,{bold:true});
text(s,'Dependency','[Resource or decision needed]',64,470,832,36);slides.push(s);

s=closingSource.duplicate();for(const i of [...s.images.items])i.delete();
sh=s.shapes.items.find(x=>x.id==='2');sh.text='Questions and Discussion';sh.position={left:290,top:233,width:620,height:100};sh.text.style={typeface:'Arial',fontSize:40,bold:true,color:'#FFFFFF',alignment:'center',autoFit:'none',insets:{top:0,left:0,bottom:0,right:0}};
s.speakerNotes.textFrame.setText('结束页。可根据需要修改为 Thank you。');slides.push(s);

for(const original of originals)original.delete();
for(let i=0;i<slides.length;i++)slides[i].moveTo(i);
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});
const candidate=path.join(TMP,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const sourceHash=createHash('sha256').update(await fs.readFile(SOURCE)).digest('hex');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:OUTPUT,pythonExecutable:PYTHON,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-bullet-geometry','--validate-heading-fit',...[7,8,11].flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:[7,8,11],fontPolicy:{basis:'reference',families:['Arial','Calibri'],referencePath:SOURCE,referenceSha256:sourceHash},verifyArtifactToolImport:true,receiptPath:path.join(TMP,'validation.json')});
console.log(JSON.stringify({slideCount:slides.length,output:OUTPUT,status:result.status??'complete'}));
const final=await PresentationFile.importPptx(await FileBlob.load(OUTPUT));
await fs.mkdir(path.join(TMP,'final-preview'),{recursive:true});
for(let i=0;i<final.slides.items.length;i++){
 const b=await final.export({slide:final.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(TMP,'final-preview',`slide-${i+1}.png`),Buffer.from(await b.arrayBuffer()));
}
