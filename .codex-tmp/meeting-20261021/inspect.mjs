import fs from 'node:fs/promises';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/runtime_helpers.mjs';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const root='D:/Downloads/Final Year Project/.codex-tmp/meeting-20261021';
await fs.mkdir(root+'/source-preview',{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load('D:/Downloads/Final Year Project/Weekly Slides/2026-10-21_Meeting.pptx'));
await fs.writeFile(root+'/source.inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,image,layout',maxChars:100000})).ndjson);
console.log('SLIDES',p.slides.items.length,'MASTERS',p.masters.items.map(x=>({id:x.id,name:x.name})), 'LAYOUTS',p.layouts.items.map(x=>({id:x.id,name:x.name,placeholders:x.placeholders.summary()})));
for(let i=0;i<p.slides.items.length;i++) {
 const s=p.slides.items[i]; console.log(i+1,s.id,s.shapes.items.map(x=>({id:x.id,name:x.name,text:String(x.text),pos:x.position,style:x.text?.style})),s.images.items.map(x=>({id:x.id,frame:x.frame})));
 await fs.writeFile(root+`/source-preview/slide-${i+1}.png`,new Uint8Array(await (await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
}
