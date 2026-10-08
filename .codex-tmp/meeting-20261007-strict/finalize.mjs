import fs from 'node:fs/promises';
import path from 'node:path';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/runtime_helpers.mjs';
import {finalizePresentation} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='D:/Downloads/Final Year Project',TMP=path.join(ROOT,'.codex-tmp/meeting-20261007-strict');
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const OUT=path.join(ROOT,'Weekly Slides/2026-10-07_WAM_Quantization_Progress_Editable_Match.pptx');
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:path.join(TMP,'candidate.pptx'),finalPath:OUT,pythonExecutable:'C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit','--require-native-table-slide','5','--require-native-table-slide','6'],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[5,6],fontPolicy:{basis:'design',families:['Arial','Times New Roman']},verifyArtifactToolImport:true,receiptPath:path.join(TMP,'final.validation.json')});
console.log(JSON.stringify({output:OUT,status:result.status??'complete'}));
const{FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const p=await PresentationFile.importPptx(await FileBlob.load(OUT));await fs.mkdir(path.join(TMP,'final-preview'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){const png=await p.export({slide:p.slides.items[i],format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'final-preview',`slide-${i+1}.png`),Buffer.from(await png.arrayBuffer()));}
for(const n of [10,11]){const s=p.slides.items[n-1];for(const im of [...s.images.items])im.delete();const png=await p.export({slide:s,format:'png',scale:1.5});await fs.writeFile(path.join(TMP,'final-preview',`editable-${n}.png`),Buffer.from(await png.arrayBuffer()));}
