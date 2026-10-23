import fs from 'node:fs/promises';
import path from 'node:path';
import {finalizePresentation} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/artifact_tool_utils.mjs';
const ROOT='D:/Downloads/Final Year Project';
const TMP=ROOT+'/.codex-tmp/meeting-20261021';
const SKILL='C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations';
const req=JSON.parse(await fs.readFile(TMP+'/requirements.json','utf8'));
const output=ROOT+'/Weekly Slides/2026-10-21_Meeting_Prepared.pptx';
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:TMP+'/candidate.pptx',finalPath:output,
 pythonExecutable:'C:/Users/Raymond/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit',...req.tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 explicitTotalSlideCount:req.slides,requiredNativeTableOwnerSlides:req.tableOwners,
 fontPolicy:{basis:'reference',families:['Arial'],referencePath:ROOT+'/Weekly Slides/2026-10-21_Meeting.pptx',referenceSha256:req.sourceHash},
 verifyArtifactToolImport:true,receiptPath:TMP+'/final.validation.json'});
console.log(JSON.stringify({output,status:result.status??'complete',slideCount:req.slides}));
