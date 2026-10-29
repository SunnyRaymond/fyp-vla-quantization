import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {finalizePresentation} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations/container_tools/artifact_tool_utils.mjs';
const skill='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const base='D:/Downloads/Final Year Project/Weekly Slides';
const ref='C:/Users/Raymond/.codex/skills/artifact-template-ntu-group-meeting/assets/reference.pptx';
process.env.RUNTIME_NODE_MODULES='C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const result=await finalizePresentation({
 workspaceDir:base,candidatePath:base+'/2026-10-28_build/candidate-v2.pptx',finalPath:base+'/2026-10-28_output/2026-10-28_Meeting_Final.pptx',
 pythonExecutable:'C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-bullet-geometry','--validate-heading-fit',...[6,7,10,11,12,13,14,15,16,17,19,20,21,22].flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:23,requiredNativeTableOwnerSlides:[6,7,10,11,12,13,14,15,16,17,19,20,21,22],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:ref,referenceSha256:createHash('sha256').update(await fs.readFile(ref)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:base+'/2026-10-28_build/validation-v2.json'
});
console.log(JSON.stringify(result));
