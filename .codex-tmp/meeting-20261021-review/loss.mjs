// Reference-matched single-page native PowerPoint loss figure.
export function drawLoss(k){
 const {text:T,rect:R,box:B,line:L,arrow:A,poly:P,math:M,fraction:F}=k;
 const navy='#061A3C',blue='#074A8B',teal='#006C63',red='#CE1737',gold='#FFBE39';
 const grad=(a,b)=>({type:'gradient',gradientKind:'linear',angleDeg:90,stops:[{offset:0,color:a},{offset:100000,color:b}]});
 const paleBlue=grad('#E6F2FF','#F5FAFF'),paleGold=grad('#FFF1D0','#FFF9ED'),paleRed=grad('#FFF0F2','#FFF8F8');
 R(6,4,553,797,{line:'#0B2B5C',radius:7,width:1.3});R(567,4,538,797,{line:'#0B2B5C',radius:7,width:1.3});R(1113,4,555,797,{line:'#0B2B5C',radius:7,width:1.3});
 T('(a) Soft-bin: preserve absolute values',17,15,531,35,{size:28,bold:true});
 T('(b) Action: temporal relations',579,15,515,35,{size:28,bold:true});
 T('(c) Latent: spatial + temporal relations',1127,15,530,35,{size:27,bold:true});
 B('Per-scalar distribution: each value is mapped to a 16-bin\nprobability distribution using a Gaussian and shared teacher\nnormalization statistics.',19,54,528,67,{fill:paleBlue,line:'#C0E0FC',radius:5,size:19});
 B('Match neighbor distributions over 3-step action windows\nto preserve temporal structure in the action sequence.',581,54,510,61,{fill:paleBlue,line:'#C0E0FC',radius:5,size:20});
 B('Match neighbor distributions over spatial and temporal neighbors\nin the latent maps to preserve spatiotemporal structure.',1127,54,528,60,{fill:paleBlue,line:'#C0E0FC',radius:5,size:19});
 // (a) Teacher and student scalars, shared calibration, and Gaussian-bin construction.
 B('BF16 teacher scalar x',26,133,257,32,{fill:grad('#D8F7EC','#EDFCF6'),line:'#7FCAB6',radius:5,size:21,bold:true});
 B('Quantized student scalar x̂',302,133,245,32,{fill:paleBlue,line:'#8ABFF4',radius:5,size:21,bold:true});
 M('e.g., x = 1.24 (BF16)',54,174,220,26,{size:21,align:'center'});
 M('e.g., x̂ = 1.08 (quantized)',326,174,221,26,{size:21,align:'center'});
 A(151,200,151,207,{width:1.5});A(421,200,421,207,{width:1.5});
 B('Shared frozen teacher statistics\nmean μ_c, scale s_c = max(σ_c, 0.02)',47,207,477,45,{fill:paleRed,line:'#FFABB9',radius:5,size:20});
 A(151,252,151,268,{width:1.5});A(421,252,421,268,{width:1.5});
 M('z =',94,277,46,32,{size:27});F('x − μ_c','s_c',139,267,74,50,{size:25});
 M('ẑ =',355,277,47,32,{size:27});F('x̂ − μ_c','s_c',402,267,77,50,{size:25});
 B('',26,318,508,157,{fill:paleBlue,line:'#C3DDFC',radius:6});
 T('Map each standardized scalar to 16 bins\nusing a Gaussian with σ = 0.375 and fixed edges.',36,329,488,41,{size:19,align:'center'});
 M('p_b(z) =',60,383,105,31,{size:26});M('∫',172,371,31,51,{size:42});
 M('e_{b+1}',202,369,53,21,{size:18});M('e_b',202,411,45,20,{size:18});
 M('N(t; z, 0.375²) dt',254,383,261,33,{size:26});
 T('15 interior edges from −3.5 to 3.5, plus two infinite tails',36,435,488,20,{size:17,align:'center'});
 M('e = [−∞, −3.5, −3.0, …, 3.0, 3.5, +∞]',35,455,491,20,{size:18,align:'center'});
 B('Teacher scalar distribution',47,482,228,30,{fill:grad('#D8F7EC','#F4FCF8'),line:'#87CFBA',radius:4,size:20,bold:true});
 B('Student scalar distribution',320,482,228,30,{fill:paleBlue,line:'#8ABFF4',radius:4,size:20,bold:true});
 function histogram(x,y,w,h,bars,color,border){
  R(x,y,w,h,{fill:'#FFFFFF',line:'#D9DEE5',width:.8});
  for(let v=1;v<=3;v++){const yy=y+h-v*h/3;L(x,yy,x+w,yy,{color:'#E3E8ED',width:.6});T((v/10).toFixed(1),x-34,yy-9,30,18,{size:17,align:'right'});}
  L(x,y,x,y+h,{width:1.2});L(x,y+h,x+w,y+h,{width:1.2});
  T('0.0',x-34,y+h-9,30,18,{size:17,align:'right'});
  const bw=w/16;
  for(let i=0;i<16;i++){const hh=bars[i]*h;R(x+i*bw+1.5,y+h-hh,bw-3,hh,{fill:grad(color,'#DCEFEB'),line:border,width:.6});L(x+(i+.5)*bw,y+h,x+(i+.5)*bw,y+h+4,{width:.7});}
  for(const [i,v]of [[0,'1'],[3,'4'],[7,'8'],[11,'12'],[15,'16']])T(v,x+i*bw-5,y+h+9,22,21,{size:17,align:'center'});
  T('Bin index',x+45,y+h+31,w-85,22,{size:18,align:'center'});
  T('Probability',x-97,y+33,116,22,{size:18,rotation:270,align:'center'});
 }
 histogram(67,528,202,85,[.006,.013,.028,.055,.10,.16,.28,.47,.70,.89,.77,.47,.22,.09,.036,.013],'#31B694','#35806E');
 histogram(338,528,201,85,[.007,.018,.042,.09,.18,.35,.60,.82,.88,.65,.37,.17,.08,.033,.014,.006],'#3E9CE0','#2E639F');
 B('',47,670,477,42,{fill:paleRed,line:'#FA8098',radius:6});
 M('L_bin = mean KL(p(z) || p(ẑ))',69,679,435,30,{size:26,color:red,align:'center'});
 B('Motor actions: 10 × 6   |   Future latents: t > 0',19,723,528,31,{fill:paleBlue,line:'#C2DDF7',radius:4,size:18.5,bold:true});
 T('Compute a 16-bin distribution for each scalar independently\nand average the KL over all scalars.',34,761,500,39,{size:18,align:'center'});
 // (b) Ten time-ordered actions followed by ordered 3-step window tokens.
 B('Ten ordered 6-D action vectors (first 10 actions)',582,133,508,29,{fill:'#E6F2FF',line:'none',size:20,bold:true,align:'left'});
 const subs=['₁','₂','₃','₄','₅','₆','₇','₈','₉','₁₀'];
 for(let i=0;i<10;i++)B('a'+subs[i],594+i*40,165,36,43,{fill:grad('#D7F2E8','#EFFAF5'),line:'#5FACA0',size:22,bold:true});
 M('Each aₜ ∈ ℝ⁶',1001,182,92,30,{size:19});
 A(825,215,825,243,{width:2});
 B('Overlapping 3-step ordered windows → 8 tokens (18-D)',583,247,508,31,{fill:'#DEEEFF',line:'#BCDFFF',radius:3,size:20,bold:true});
 const windows=[['w₁','(a₁, a₂, a₃)',592],['w₂','(a₂, a₃, a₄)',700],['w₃','(a₃, a₄, a₅)',808],['w₈','(a₈, a₉, a₁₀)',981]];
 for(const [label,formula,x]of windows){R(x,290,102,55,{fill:paleBlue,line:'#4488C8',width:1});T(label,x+8,298,85,23,{size:21,bold:true,align:'center'});M(formula,x+2,323,98,21,{size:18,align:'center'});}
 T('…',924,300,42,30,{size:28,align:'center'});
 M('Each token uᵢ ∈ ℝ¹⁸ (concatenate 3 actions)',607,361,470,26,{size:21,align:'center'});
 B('Relation matrix (8 tokens)',583,395,268,31,{fill:'#DFEEFF',line:'#C0DFFF',radius:3,size:20,bold:true});
 B('Neighbor probabilities',865,395,228,31,{fill:'#DFEEFF',line:'#C0DFFF',radius:3,size:20,bold:true});
 const mx=618,my=454,cw=21,ch=20.5;
 const shades=['#346CA8','#467FB9','#5D93C7','#7DADDA','#91BBDF','#5D99CD','#2D6FAD','#1F5F9D'];
 for(let i=0;i<8;i++){
  T(String(i+1),mx+i*cw,my-24,cw,22,{size:17,align:'center'});T(String(i+1),mx-27,my+i*ch+1,22,20,{size:17,align:'center'});
  for(let j=0;j<8;j++){
   const fill=i===j?'#A9B4C1':shades[(Math.abs(i-j)+Math.min(i,j)*2)%8];
   R(mx+j*cw,my+i*ch,cw,ch,{fill,line:'#C5D7E5',width:.35});
   if(i===j)T('×',mx+j*cw,my+i*ch-1,cw,ch,{size:18,align:'center',color:'#243647'});
  }
 }
 M('dᵢⱼ',807,438,34,23,{size:19,align:'center'});
 R(800,464,15,137,{fill:{type:'gradient',gradientKind:'linear',angleDeg:90,stops:[{offset:0,color:'#1E609E'},{offset:100000,color:'#D2E6F4'}]},line:'#B6CADB',width:.6});
 T('high',823,466,29,21,{size:16});T('low',823,585,29,21,{size:16});
 T('Diagonal masked (no self-relation)',591,635,267,22,{size:18,align:'center'});
 T('All other windows are neighbors',591,659,267,22,{size:18,align:'center'});
 R(865,432,228,257,{fill:'#F8FBFF',line:'#C3DDF3',width:.75,radius:4});
 M('dᵢⱼ =',880,461,55,31,{size:24});F('1','18',936,452,25,51,{size:20});
 M('∑',963,452,39,45,{size:38});M('18',973,438,29,20,{size:16});M('k=1',960,496,43,20,{size:16});
 M('(uᵢₖ − uⱼₖ)²',997,464,91,32,{size:23});
 M('p(j | i) =',875,538,89,34,{size:25});
 F('exp(−dᵢⱼ / τₐ)','∑ₖ≠ᵢ exp(−dᵢₖ / τₐ)',961,521,126,61,{size:21});
 M('for j ≠ i',936,597,139,27,{size:23,align:'center'});
 M('τₐ: frozen temperature',880,623,201,25,{size:21,align:'center'});
 T('(teacher and student share τₐ)',879,649,203,25,{size:18,align:'center'});
 B('',588,702,506,59,{fill:paleRed,line:'#F889A0',radius:6});
 M('L_temp,a = mean_i KL(p^T(· | i) || p^S(· | i))',599,719,485,35,{size:24,color:red,align:'center'});
 T('All other windows are neighbors (diagonal masked).',594,775,491,24,{size:18,align:'center'});
 // (c) Separate camera grids; one camera has the spatial/temporal edge illustration.
 function cameraBox(x,y,title,withEdges){
  R(x,y,344,252,{fill:grad('#FFF5DC','#FFFDF8'),line:'#F3D18C',width:1,radius:7});
  B(title,x,y,344,34,{fill:paleGold,line:'none',radius:5,size:21,bold:true,align:'left'});
  T('Time 1 (future)',x+21,y+44,145,24,{size:20,bold:true,align:'center'});
  T('Time 2 (future)',x+192,y+44,145,24,{size:20,bold:true,align:'center'});
  const gx=x+24,gy=y+66,step=20.5;
  for(let t=0;t<2;t++)for(let r=0;r<7;r++)for(let c=0;c<7;c++){
   const focus=withEdges&&r===3&&c===3;
   const adjacent=withEdges&&t===0&&Math.max(Math.abs(r-3),Math.abs(c-3))===1;
   R(gx+t*171+c*step,gy+r*step,19.2,19.2,{fill:focus?'#FF8156':adjacent?'#FFD251':gold,line:focus?'#D74425':'#F0A71C',width:focus?1.4:.55});
  }
  if(withEdges){
   const cx=gx+3*step+9.5,cy=gy+3*step+9.5;
   for(let dr=-1;dr<=1;dr++)for(let dc=-1;dc<=1;dc++)if(dc||dr)A(cx,cy,cx+dc*44,cy+dr*44,{width:1.5,color:'#153B6A'});
   A(cx+13,cy,cx+171-12,cy,{dash:true,start:true,width:1.6,color:'#153B6A'});
  }
 }
 cameraBox(1127,126,'Camera 1 (view 1)',true);cameraBox(1127,390,'Camera 2 (view 2)',false);
 R(1480,133,176,254,{fill:'#FFFFFF',line:'#CBDFEE',width:.8,radius:5});
 R(1493,147,25,25,{fill:gold,line:'#E8A321',width:.9});T('Latent token\n(48-D)',1527,145,118,44,{size:19});
 R(1493,190,25,23,{fill:'#FF8156',line:'#F35A36',width:.9});T('Selected location',1527,191,119,28,{size:18});
 A(1490,240,1524,240,{start:true,dash:true,width:1.5});T('Temporal edge\n(same position,\nsame camera only)',1530,228,118,64,{size:18});
 A(1507,288,1507,329,{start:true,width:1.5});A(1488,309,1525,309,{start:true,width:1.5});
 T('8 spatial neighbors\n(same time)',1530,300,119,43,{size:18});
 L(1494,351,1513,370,{color:'#F31627',width:2.4});L(1513,351,1494,370,{color:'#F31627',width:2.4});
 T('No cross-camera\nedges',1530,350,118,41,{size:18});
 B('Each view’s original latent\nmap is 14 × 14, pooled\nwith 2 × 2 average pooling\nto 7 × 7 (separately\nfor each view).',1481,436,175,107,{fill:'#FFFFFF',line:'#C3DFF6',radius:5,size:18});
 B('2 future times × 2 views × 7 × 7 = 196 tokens, 48-D',1127,613,527,37,{fill:paleGold,line:'#EDCD8E',radius:5,size:21,bold:true});
 B('',1128,664,527,55,{fill:paleRed,line:'#F88CA0',radius:5});
 M('L_st,z = mean_i KL(p^T(· | i) || p^S(· | i))',1139,677,505,35,{size:25,color:red,align:'center'});
 T('8 local spatial neighbors + same-position temporal edge (same camera only).\nNo cross-camera edges.',1124,737,532,46,{size:18,align:'center'});
 // Bottom equation and the two original boundary notes.
 B('',94,812,1514,56,{fill:paleRed,line:'#F66E8F',radius:8});
 M('L = αₐ (L_bin,a + βₐ L_temp,a) + λ α_z (L_bin,z + β_z L_st,z)',254,824,1192,42,{size:36,align:'center'});
 B('T: frozen teacher. S: quantized student. Shared frozen statistics and temperatures.\nTrain D and clipping ratios. Soft-bin anchors values, relation KL constrains structure.',10,880,819,48,{fill:paleBlue,line:'#C5DFF6',radius:5,size:17.5,align:'left',pad:7});
 B('Evidence: fixed-input fidelity. Rollout prediction and closed-loop gains await validation.',868,880,797,48,{fill:paleBlue,line:'#C5DFF6',radius:5,size:18,align:'center',pad:8});
 return k.count;
}
