// Faithful native reconstruction of the supplied method image; all coordinates are image pixels.
export function drawMethod(k){
 const {text:T,rect:R,box:B,line:L,arrow:A,poly:P,nativeArrow:NA,custom:C,math:M,grid:G}=k;
 const navy='#061A3C',blue='#074A8B',teal='#00696D',red='#B10818';
 const grad=(a,b)=>({type:'gradient',gradientKind:'linear',angleDeg:90,stops:[{offset:0,color:a},{offset:100000,color:b}]});
 const blueFill=grad('#B9E4FF','#DFF3FF'),redFill=grad('#FFC1C1','#FFE8E2'),goldFill=grad('#FFE6B3','#FFF3DD');
 R(4,4,706,669,{radius:8,line:navy,width:1.7});R(718,4,478,669,{radius:8,line:navy,width:1.7});R(1203,4,465,669,{radius:8,line:navy,width:1.7});
 T('(a) Latent + action supervision',17,14,678,37,{size:29,bold:true});
 T('(b) Microscaling additive VQ weights',731,14,454,37,{size:26,bold:true});
 T('(c) Per-channel activation scaling + clipping',1213,16,447,33,{size:24,bold:true});
 // Panel a: matching teacher and student pathways.
 R(157,57,393,263,{fill:grad('#D9EFFF','#EEF8FF'),line:'none',radius:7});
 R(157,325,393,286,{fill:grad('#FFE2DC','#FFF0EA'),line:'none',radius:7});
 for(const ang of [0,Math.PI/3,Math.PI*2/3]){const dx=11*Math.cos(ang),dy=11*Math.sin(ang);L(180-dx,77-dy,180+dx,77+dy,{color:'#0785D0',width:2});}
 T('BF16 teacher (frozen)',199,66,335,26,{size:23,bold:true,color:blue});
 C([[12,0],[18,13],[26,21],[24,29],[19,35],[9,35],[2,28],[2,19],[7,10],[8,24],[13,17]],167,330,28,36,{fill:'#FF9C23',line:'#EF711C'});
 C([[9,0],[12,10],[18,14],[15,22],[8,24],[3,19],[4,12]],173,341,19,25,{fill:'#FFD785',line:'none'});
 T('Quantized WAM (student)',199,336,345,26,{size:23,bold:true,color:red});
 T('614 target Linear layers',343,359,199,24,{size:21,bold:true,color:'#C11B14'});
 L(343,382,542,382,{color:'#C11B14',width:1.3});
 function dit(x,y,kind,student){
  C([[0,0],[138,26],[138,68],[0,94]],x,y,138,94,{fill:student?redFill:blueFill,line:student?'#A92024':'#0B579E',width:1.3});
  T(kind+' DiT\n'+(student?'(quantized)':'(BF16)'),x+16,y+25,109,53,{size:19,bold:true,align:'center'});
 }
 dit(205,103,'Video',false);dit(397,106,'Action',false);dit(205,378,'Video',true);dit(397,382,'Action',true);
 A(343,149,397,149,{width:1.9});A(343,425,397,425,{width:1.9});
 B('',12,172,128,247,{fill:grad('#F0F2F5','#E8EBEF'),line:'#63849E',radius:5});
 T('Observation\n+ language\n+ proprioception',23,208,108,67,{size:19,bold:true,align:'center'});
 M('o, l, p',37,280,83,31,{size:24,align:'center'});
 M('{ x₀, …, xT',20,314,118,43,{size:26});T('text',35,355,80,27,{size:20});M('q, q̂',34,382,83,28,{size:25});
 P([[140,230],[169,230],[169,149],[205,149]],{width:1.7});
 P([[140,403],[169,403],[169,430],[205,430]],{width:1.7});
 function tokenRow(x,y,fill,stroke){for(let i=0;i<4;i++)R(x+i*16,y,9,39,{fill,line:stroke,width:1});}
 tokenRow(237,196,'#EFB956','#C58823');tokenRow(430,196,'#71C9CA','#2E8F97');
 tokenRow(237,475,'#F6BC54','#C58823');tokenRow(430,475,'#71C9CA','#2E8F97');
 T('Future latent tokens Zᵀ',187,242,221,24,{size:19,bold:true});
 T('(from future frames 1 … T)',190,265,212,21,{size:17});
 T('Action chunk aᵀ',409,242,139,23,{size:20,bold:true});
 T('(6 motor dims, gripper excluded)',373,265,176,23,{size:17});
 T('Future latent tokens Zˢ',188,526,215,24,{size:19,bold:true});
 T('(from student’s own generation)',186,549,201,23,{size:17,align:'center'});
 T('Action chunk aˢ',409,526,139,24,{size:20,bold:true});
 T('(6 motor dims, gripper excluded)',391,549,156,23,{size:17,align:'center'});
 function brace(x,y,w,label){P([[x,y],[x,y+7],[x+w,y+7],[x+w,y]],{end:false,width:1.2});T(label,x-7,y+14,w+15,25,{size:17,align:'center'});}
 brace(188,285,198,'T tokens (exclude frame 0)');brace(399,285,121,'H tokens');
 brace(188,568,198,'T tokens (exclude frame 0)');brace(399,568,121,'H tokens');
 P([[303,220],[402,220],[402,192]],{width:1.3});
 T('Video K/V',325,190,84,20,{size:17,italic:true,align:'center'});T('(from teacher Zᵀ)',307,210,97,20,{size:15,align:'center'});
 P([[303,500],[402,500],[402,470]],{width:1.3});
 T('Video K/V',324,470,82,20,{size:17,italic:true,align:'center'});T('(from student Zˢ)',307,490,99,20,{size:15,align:'center'});
 // Native dashed connectors distinguish offline comparisons from inference.
 P([[521,149],[560,149],[560,125],[582,125]],{dash:true,width:1.5});
 P([[521,425],[560,425],[560,441],[582,441]],{dash:true,width:1.5});
 L(560,125,560,441,{dash:true,width:1.2});
 B('',582,94,122,175,{fill:goldFill,line:'#F0D297',radius:5});
 T('Future\nlatent loss',592,102,102,48,{size:20,bold:true,color:'#76450F',align:'center'});
 M('L_z = ‖Z^T_{1:T} −',590,164,108,30,{size:24,align:'center'});M('Z^S_{1:T}‖²₂',601,191,98,31,{size:24,align:'center'});
 T('(exclude\nframe 0)',600,225,92,39,{size:18,align:'center'});
 B('',582,334,122,207,{fill:grad('#D9F2F4','#EDF9FA'),line:'#98D7DE',radius:5});
 T('First-10\nmotor action\nloss',591,347,105,64,{size:20,bold:true,align:'center'});
 M('L_a = ‖a^T_{1:10} −',589,420,109,31,{size:23,align:'center'});M('a^S_{1:10}‖²₂',599,450,100,31,{size:23,align:'center'});
 T('(6 motor dims\nonly, gripper\nexcluded)',590,485,104,52,{size:18,align:'center'});
 // Panel b: B32 block, vector codes and additive decode.
 T('Transform a weight row and split into groups of 32 scalars',738,53,445,24,{size:20,align:'center'});
 T('(each vector has d = 4)',749,75,426,24,{size:20,align:'center'});
 T('Transformed weight row  w',766,101,382,23,{size:20});
 G(775,123,1,16,367,21,{fill:'#F3F4F5',line:'#58697C',width:.75,gap:0});
 G(775,154,1,8,157,20,{fill:'#A3DBED',line:'#2F89AB',width:.8,gap:0});
 G(939,154,1,8,148,20,{fill:'#A1DAD7',line:'#378C8E',width:.8,gap:0});
 T('…',1118,143,48,32,{size:28,align:'center'});
 P([[775,178],[775,184],[930,184],[930,178]],{end:false,width:1.1,color:blue});
 T('Group g (32 scalars)',775,188,166,25,{size:19,bold:true,color:blue});
 T('Group g + 1',975,188,119,25,{size:19});
 R(737,222,422,62,{fill:'#FFFFFF',line:'#5C6D82',width:1,dash:true,radius:4});
 T('32 scalars = 8 vectors (d = 4)',750,227,396,23,{size:19,align:'center'});
 for(const x of [765,860,974])G(x,257,1,4,81,18,{fill:'#F1F3F4',line:'#627386',width:.65});
 T('…',944,252,24,27,{size:24});T('…',1102,252,26,27,{size:24});
 P([[776,155],[758,155],[740,222]],{dash:true,color:'#246EAF',width:1.2});
 P([[1086,176],[1127,192],[1150,221]],{dash:true,color:'#69748A',width:1.1,end:false});
 function codebook(x,n,fill,stroke){
  R(x,294,215,185,{fill,line:stroke,width:1,radius:4});
  T('Codebook '+n,x+10,299,195,25,{size:21,bold:true,color:n===1?blue:teal,align:'center'});
  M('C'+(n===1?'₁':'₂')+' ∈ ℝ²⁵⁶×⁴, BF16',x+11,323,192,26,{size:22,align:'center'});
  T('8 indices',x+12,348,83,21,{size:18,bold:true});T('i'+(n===1?'₁':'₂')+' (8-bit)',x+18,369,82,21,{size:17});
  for(const [j,y]of [[1,384],[2,408],[8,448]])B('i'+(n===1?'₁':'₂')+','+j,x+22,y,41,22,{fill:'#F6FAFE',line:'#64778C',size:16,font:'Times New Roman'});
  T('⋮',x+31,430,25,19,{size:20});
  T('8 codewords',x+104,355,100,22,{size:17,bold:true,align:'center'});T('(each 4 scalars)',x+100,376,106,21,{size:16,align:'center'});
  T('Lookup',x+72,394,55,19,{size:16,bold:true});A(x+73,418,x+123,418,{color:n===1?blue:teal,width:1.4});
  for(const y of [390,416,442,465])G(x+138,y,1,4,53,14,{fill:n===1?'#8DC7EE':'#87CDCC',line:stroke,width:.75});
 }
 codebook(732,1,'#F3F9FF','#6BB7EB');codebook(975,2,'#EFFAF8','#41A0A5');
 T('+',948,394,25,39,{size:32,bold:true,color:blue});
 B('',733,490,447,43,{fill:goldFill,line:'#F3DCA8',radius:4});
 M('Ŵ[g] = r_row × 2ᵉᵍ × (C₁[i₁] + C₂[i₂])',742,498,429,32,{size:25,align:'center'});
 A(783,558,800,534,{color:blue,width:1.2});A(897,557,913,534,{color:'#0776A5',width:1.2});A(1020,558,1020,534,{color:teal,width:1.2});A(1138,558,1138,534,{color:navy,width:1.2});
 T('BF16 row gain',733,556,100,20,{size:17,bold:true,color:blue,align:'center'});M('r_row',756,579,57,21,{size:19,color:blue,align:'center'});
 T('B32 power-of-two\nscales 2ᵉᵍ',841,555,127,43,{size:17,bold:true,color:'#0877A6',align:'center'});
 T('2 × 8-bit indices /\n4 weights = 4 core bpw',975,555,126,43,{size:17,bold:true,color:teal,align:'center'});
 T('~4.27 bpw\nincluding\nmetadata',1115,551,76,55,{size:18,bold:true,align:'center'});
 B('',733,613,447,47,{fill:redFill,line:'#F49BA3',radius:5});
 T('Indices and B32 exponents are frozen',744,621,425,22,{size:20,bold:true,color:red,align:'center'});
 T('Codebooks and row gains calibrated with action + latent supervision',743,643,428,18,{size:16,align:'center'});
 // Panel c: scaled activation and compensated stored weights.
 B('Activation path (input X)',1217,60,237,37,{fill:'#DDEFFF',line:'none',size:21,bold:true,color:'#0C3873',radius:3});
 M('X',1219,158,26,31,{size:29});
 B('Smooth\n(α = 0.5)\n+ H128',1269,126,91,90,{fill:blueFill,line:'#1269AC',size:20,bold:true});
 B('÷ D',1401,143,51,55,{fill:blueFill,line:'#4088B5',size:24,font:'Times New Roman',italic:true});
 B('Dynamic\nsigned A4',1473,127,87,89,{fill:grad('#76D1D5','#B4E4E3'),line:'#106C7A',size:19,bold:true});
 B('BF16\ndense\nLinear',1580,126,74,90,{fill:blueFill,line:'#1666AC',size:20,bold:true});
 A(1244,171,1268,171,{width:1.5});A(1361,170,1399,170,{width:1.4});M('Z',1371,144,24,24,{size:23});
 A(1453,170,1472,170,{width:1.4});A(1561,170,1579,170,{width:1.4});
 T('D: one learned positive scale per physical input channel',1227,230,425,26,{size:20});
 M('v = Z / D',1230,283,122,35,{size:25});
 B('',1353,256,300,105,{fill:grad('#E1F5FC','#F1FBFE'),line:'#B6E0EE',radius:5});
 T('Per-token dynamic signed A4 (per Linear)',1363,266,280,25,{size:18,align:'center'});
 M('sₜ = r_l × max |vₜ| / 7',1383,296,246,29,{size:24,align:'center'});
 M('q = clamp(round(v / sₜ), −7, 7)',1368,327,268,28,{size:22,align:'center'});
 T('r_l: one clipping ratio per Linear',1360,371,293,23,{size:19,color:red});
 T('(distinct from per-channel D)',1387,394,260,22,{size:19,align:'center'});
 B('Weight path',1217,425,146,40,{fill:'#FDE7E5',line:'none',size:23,bold:true,color:red,radius:4});
 B('Decode VQ\n(frozen)',1217,477,128,70,{fill:redFill,line:'#C94E64',size:22,bold:true,color:red});
 M('Ŵ',1362,488,36,31,{size:28});
 B('Ŵ × D',1407,484,114,57,{fill:goldFill,line:'#EBC778',size:27,font:'Times New Roman',italic:true});
 B('BF16\ndense\nLinear',1561,469,92,88,{fill:blueFill,line:'#156CB0',size:21,bold:true,color:blue});
 A(1346,512,1406,512,{width:1.6});A(1522,512,1560,512,{width:1.6});
 B('',1217,571,436,85,{fill:grad('#DBF0F0','#F4FCFC'),line:'#95D0D1',radius:6});
 T('Training refinement (activation only)',1226,578,416,24,{size:21,bold:true,color:teal});
 T('• Keep VQ weights frozen (indices, B32 exponents, codebooks, row gains)',1228,605,415,20,{size:17});
 T('• Learn per-channel scales D and clipping ratio r_l',1228,630,414,20,{size:18});
 // Bottom strip: original offline training loop with native multi-turn arrows.
 R(4,684,1664,252,{line:navy,width:1.6,radius:8});
 T('Offline training loop (action + latent supervision)',18,695,742,36,{size:27,bold:true});
 T('Offline gradients (no gradient to frozen components)',806,708,540,22,{size:17});
 A(1391,708,1454,708,{color:'#003C82',width:2.2});T('Forward data',1462,698,181,22,{size:17});
 A(1391,730,1454,730,{dash:true,width:1.5});T('Offline gradient / parameter update',1462,722,193,35,{size:16});
 B('BF16 teacher\n(frozen)',18,770,154,53,{fill:blueFill,line:'#4C9ED7',size:20,bold:true,color:blue,radius:4});
 B('Quantized WAM\n(student)',18,840,154,52,{fill:redFill,line:'#F06F80',size:20,bold:true,color:red,radius:4});
 B('Forward pass\non training data\n(action + latent supervision)',206,784,216,97,{fill:'#F8FBFF',line:'#17619B',size:21,radius:4});
 A(173,797,205,797,{width:1.8});A(173,865,205,865,{width:1.8});
 B('Future latent loss  Lz',451,777,238,43,{fill:goldFill,line:'#EDBE67',size:21,color:'#845217',radius:4});
 B('First-10 motor action loss  La',451,839,238,44,{fill:grad('#D7EFF1','#F0FAFA'),line:'#72BFCA',size:20,bold:true,color:'#004F65',radius:4});
 A(423,798,450,798,{width:1.7});A(423,861,450,861,{width:1.7});
 P([[690,799],[710,799],[710,826],[732,826]],{width:1.5});P([[690,861],[710,861],[710,826],[732,826]],{width:1.5});
 T('Total loss',755,795,128,25,{size:23,bold:true});M('L = Lz + La',745,825,143,40,{size:29});
 B('Calibrate codebooks /\nrow gains\n(section b)',943,764,215,69,{fill:redFill,line:'#F18591',size:22,bold:true,color:red,radius:5});
 B('Refine per-channel D /\nclipping ratio\n(section c)',1188,764,212,69,{fill:grad('#ACE3E4','#D9F1F1'),line:'#84CDD1',size:22,bold:true,color:'#004D61',radius:5});
 P([[886,826],[915,826],[915,791],[942,791]],{dash:true,width:1.5});
 T('Update: codebooks C₁, C₂\nand row gains r_row',960,842,188,39,{size:18,align:'center'});
 T('(indices and B32 exponents remain frozen)',941,883,219,23,{size:17,align:'center'});
 T('Update: per-channel scales D\nand per-Linear clipping ratio r_l',1195,842,202,40,{size:18,align:'center'});
 T('(VQ weights remain frozen)',1212,884,177,22,{size:17,align:'center'});
 P([[1053,763],[1053,746],[596,746],[596,776]],{dash:true,width:1.3});
 P([[1293,763],[1293,728],[262,728],[262,783]],{dash:true,width:1.3});
 T('Current execution: decoded BF16 weights + fake A4 + BF16 KV; native kernel remains future work.',1054,914,598,20,{size:16,italic:true,color:'#34526A'});
 return k.count;
}
