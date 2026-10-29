import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
import {repair} from './author.mjs';
import {installNode} from '@oai/granola/node';
installNode();
process.on('uncaughtException',e=>{console.error(e.message);process.exit(1);});
process.on('unhandledRejection',e=>{console.error(e?.message??e);process.exit(1);});
const root='D:/Downloads/Final Year Project/Weekly Slides/2026-10-28_build';
const input='C:/Users/Raymond/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Roaming/Codex/web/Codex/artifact-sessions/exports/latest/2a6b22b0a3a1e8db840ead81e65efdb561b692ef1aa8cee512f0fea30f0718ca/2026-10-28_Meeting_Draft.pptx';
const p=await Presentation.load(input);console.log('loaded',p.slides.items.length);
await repair({presentation:p});console.log('repaired');
await (await PresentationFile.exportPptx(p)).save(path.join(root,'candidate.pptx'));console.log('exported');
await fs.mkdir(path.join(root,'review'),{recursive:true});
for(const i of [0,2,3,4,6,7,19,20,21]){const blob=await p.export({slide:p.slides.getItem(i),format:'png',scale:1});await fs.writeFile(path.join(root,'review',`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));console.log('rendered',i+1);}
