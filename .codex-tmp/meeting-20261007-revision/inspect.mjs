import fs from 'node:fs/promises';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load('D:/Downloads/Final Year Project/Weekly Slides/2026-10-07_WAM_Quantization_Progress.pptx'));
const rows=(await p.inspect({kind:'slide,shape,textbox,table',maxChars:180000})).ndjson.split('\n').filter(Boolean).map(s=>JSON.parse(s));
await fs.writeFile('D:/Downloads/Final Year Project/.codex-tmp/meeting-20261007-revision/source.inspect.json',JSON.stringify(rows,null,2));
console.log(p.help('*',{search:'slide.moveTo|slide.duplicate',include:['index','examples','notes'],maxChars:13000}).ndjson);

