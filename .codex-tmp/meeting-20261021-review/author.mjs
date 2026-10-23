import fs from 'node:fs/promises';
import {importRuntimeModule} from 'file:///C:/Users/Raymond/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.11052/skills/presentations/container_tools/runtime_helpers.mjs';
import {createKit} from './kit.mjs';
import {drawMethod} from './method.mjs';
import {drawLoss} from './loss.mjs';
const {FileBlob,PresentationFile}=await importRuntimeModule('@oai/artifact-tool');
const dir='D:/Downloads/Final Year Project/.codex-tmp/meeting-20261021-review';
const p=await PresentationFile.importPptx(await FileBlob.load(dir+'/source-current.pptx'));
const original=[...p.slides.items],method=original[1],loss=original[8];
for(const [s,draw]of [[method,drawMethod],[loss,drawLoss]]){
 for(const a of [...s.shapes.items])a.delete();for(const a of [...s.images.items])a.delete();for(const a of [...s.tables.items])a.delete();for(const a of [...s.charts.items])a.delete();
 const k=createKit(s);draw(k);console.log(draw.name,'native objects',k.count);
 s.speakerNotes.textFrame.setText(draw===drawMethod?
 '当前实现：latent+action-aware Microscaling additive VQ，随后进行 per-input-channel D 和 per-Linear clipping ratio 的 refinement。沿用原第21页ImageGen设计，修正生成图中的不规范公式与输入标签；所有文字、公式、矩阵与箭头为PowerPoint原生可编辑对象，全部Times New Roman。Source: Weekly Slides/2026-10-21_assets/method-imagegen-reference.png。L_a为first10x6 motor MSE；L_z为生成future latent NMSE，observed frame0排除；L=L_a+lambda L_z，lambda=rho*c从初始calibration冻结。Student使用自己生成的future，不注入teacher future。Per-token动态A4，gripper不参与监督。当前执行decodedBF16+fakeA4+BF16KV，非nativeVQkernel。':
 '本页严格按原第22页ImageGen设计重建Soft-bin + temporal/spatial relational KL。全部文字、数学文字、矩阵、柱图、网格和箭头均为PowerPoint原生可编辑对象。Source: Weekly Slides/2026-10-21_assets/loss-imagegen-reference.png。柱图和relationmatrix颜色为原图概念示意，不是新增实验观测。实际Gaussian CDF binning、action 3-step windows、两camera分开pooling和spatiotemporal邻接保持原实现。结构KL不估计完整action-latent联合密度；闭环结果仍待验证。');
}
for(const s of original)if(s!==method&&s!==loss)s.delete();method.moveTo(0);loss.moveTo(1);
await(await PresentationFile.exportPptx(p)).save(dir+'/authored.pptx');
console.log('authored',p.slides.items.length,'slides');
