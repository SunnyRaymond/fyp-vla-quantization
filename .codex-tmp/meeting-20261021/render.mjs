import fs from 'node:fs/promises';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/runtime_helpers.mjs';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const ROOT='D:/Downloads/Final Year Project/.codex-tmp/meeting-20261021';
const input=process.argv[2]??ROOT+'/candidate.pptx';
const out=process.argv[3]??ROOT+'/preview';
await fs.mkdir(out,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(input));
for(let i=0;i<p.slides.items.length;i++){
 if(process.argv.length>4&&!process.argv.slice(4).includes(String(i+1)))continue;
 await fs.writeFile(out+`/slide-${i+1}.png`,new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1.5})).arrayBuffer()));
 console.log('rendered',i+1);
}
await fs.writeFile(out+'/inspect.ndjson',(await p.inspect({kind:'slide,textbox,shape,table,image',maxChars:1000000})).ndjson);
