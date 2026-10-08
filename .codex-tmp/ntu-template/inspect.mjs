import fs from 'node:fs/promises';
import { importRuntimeModule } from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
const { FileBlob, PresentationFile } = await importRuntimeModule('@oai/artifact-tool');
const p = await PresentationFile.importPptx(await FileBlob.load('D:/Downloads/Presentation_GuoYichen.pptx'));
const snap = await p.inspect({kind:'slide,textbox,shape,image,layout',maxChars:100000});
await fs.writeFile('D:/Downloads/Final Year Project/.codex-tmp/ntu-template/source-inspect.ndjson',snap.ndjson);
console.log('MASTERS',p.masters.items.map(m=>({id:m.id, name:m.name, shapes:m.shapes.items.map(s=>({id:s.id,name:s.name,text:String(s.text),position:s.position})),images:m.images.items.map(i=>({id:i.id,frame:i.frame}))})));
console.log('LAYOUTS',p.layouts.items.map(l=>({id:l.id,name:l.name,placeholders:l.placeholders.summary()})));
for (const n of [1,6,7,10,13]) {
 const s=p.slides.items[n-1];
 console.log('SLIDE',n,s.id,'SHAPES',s.shapes.items.map(x=>({id:x.id,name:x.name,text:String(x.text),position:x.position,style:x.text?.style})),'IMAGES',s.images.items.map(x=>({id:x.id,frame:x.frame,alt:x.alt})));
}
