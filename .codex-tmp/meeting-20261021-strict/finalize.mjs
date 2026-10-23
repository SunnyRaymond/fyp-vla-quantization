import fs from 'node:fs/promises';
import path from 'node:path';
import {finalizePresentation} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='D:/Downloads/Final Year Project',TMP=ROOT+'/.codex-tmp/meeting-20261021-strict';
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
const tableOwners=[3,4,5,6,7,8,10,14,15,16];
await fs.mkdir(TMP+'/output',{recursive:true});
const output=TMP+'/output/2026-10-21_Meeting_Prepared.pptx';
await finalizePresentation({workspaceDir:ROOT,candidatePath:TMP+'/candidate.pptx',finalPath:output,
 pythonExecutable:'C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:tableOwners,
 fontPolicy:{basis:'design',families:['Arial','Arial Narrow','Times New Roman']},verifyArtifactToolImport:true,receiptPath:TMP+'/final.validation.json'});
console.log('Validated output:',output);
