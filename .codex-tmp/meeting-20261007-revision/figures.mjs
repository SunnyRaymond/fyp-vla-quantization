// Editable PowerPoint figures authored directly. No Draw.io or drawing plugin.
const P={navy:'#253D70',blue:'#4D7CAC',blueLight:'#EAF0F8',gold:'#B08227',goldLight:'#F9F2DE',teal:'#24827F',tealLight:'#E8F3F1',red:'#BC5560',redLight:'#FAECEE',ink:'#233044',gray:'#6B7380',line:'#C9D0D9'};
const shades={blue:['#F3F6FB','#DEE8F3','#B8CEE6','#84ABD2','#4D7CAC'],gold:['#FFFCF3','#F7EBCB','#ECD29A','#D7B564','#B08227'],teal:['#F1F8F7','#D5EBE6','#A2D0C6','#61AAA0','#24827F'],red:['#FCF3F4','#F7DCE1','#ECA7B2','#D87F8E','#BC5560']};
let seq=0;
export function canvas(s,ox=48,oy=84,{font='Times New Roman'}={}){
 function S(x,y,w,h,fill='none',stroke='none',lw=.8,geom='rect',dash=false){return s.shapes.add({name:`Figure element ${++seq}`,geometry:geom,position:{left:ox+x,top:oy+y,width:w,height:h},fill,line:{fill:stroke,width:lw,style:dash?'dashed':'solid'}});}
 function T(text,x,y,w,h,{size=18,color=P.ink,bold=false,align='left',italic=false,typeface=font,middle=false}={}){const q=S(x,y,w,h);q.text=text;q.text.style={typeface,fontSize:size,bold,italic,color,alignment:align,verticalAlignment:middle?'middle':'top',autoFit:'none',wrap:'square',insets:{top:0,bottom:0,left:0,right:0}};return q;}
 function L(x1,y1,x2,y2,{color=P.ink,width=1.15,arrow=false,dash=false,kind='straight'}={}){const a=S(x1-.25,y1-.25,.5,.5),b=S(x2-.25,y2-.25,.5,.5);return s.shapes.connect(a,b,{kind,fromSide:'right',toSide:'left',line:{fill:color,width,style:dash?'dashed':'solid'},...(arrow?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});}
 function A(x1,y1,x2,y2,opt={}){return L(x1,y1,x2,y2,{...opt,arrow:true});}
 function M(x,y,r,c,cell,{palette='blue',pattern,groups=null,outline=true}={}){for(let i=0;i<r;i++)for(let j=0;j<c;j++){const pal=shades[groups?.[j]??palette],v=pattern?pattern(i,j):(i*3+j*2+1)%4;S(x+j*cell,y+i*cell,cell,cell,pal[Math.max(0,Math.min(4,v))],'#FFFFFF',.5);}if(outline)S(x,y,c*cell,r*cell,'none',P.gray,.75);return{left:x,top:y,width:c*cell,height:r*cell};}
 function D(x,y,color=P.navy,r=4.3){return S(x-r,y-r,2*r,2*r,color,color,.8,'ellipse');}
 function E(x,y,rx,ry,{fill='none',stroke=P.teal,width=1.15,dash=false}={}){return S(x-rx,y-ry,2*rx,2*ry,fill,stroke,width,'ellipse',dash);}
 function panel(text,x,w){T(text,x,0,w,31,{size:21,bold:true,color:P.navy});}
 return{S,T,L,A,M,D,E,panel,P};
}

export function methodA(s){
 const g=canvas(s,48,88),{S,T,L,A,M,panel,P}=g;
 panel('(a) Paired conditioning',0,285);panel('(b) Vector codebook',317,274);panel('(c) ActionDiT inference',625,255);
 L(299,0,299,372,{color:P.line,dash:true});L(608,0,608,372,{color:P.line,dash:true});
 T('Activations X',8,46,130,26,{bold:true});T('Conditioned X′',167,46,126,26,{bold:true});
 M(12,82,4,6,14,{pattern:(r,c)=>c===4?4:(r+c)%3});M(195,82,4,6,14,{pattern:(r,c)=>(r*2+c)%3});
 A(105,110,183,110,{color:P.blue,width:1.7});T('D⁻¹H',113,78,67,27,{size:22,align:'center',color:P.blue});
 T('SmoothQuant scaling\n+ Hadamard rotation',49,151,235,47,{size:17,color:P.gray,align:'center'});
 T('Weights W',8,207,126,25,{bold:true});T('Weights W′',167,207,126,25,{bold:true});
 M(12,240,4,6,14,{palette:'gold'});M(195,240,4,6,14,{palette:'gold',pattern:(r,c)=>(r+c+2)%5});
 A(105,268,183,268,{color:P.gold,width:1.7});T('DH',118,237,58,27,{size:22,align:'center',color:P.gold});
 T('X′(W′)ᵀ = XWᵀ',8,323,283,29,{size:23,bold:true,align:'center'});
 T('Exact before quantization',15,353,275,23,{size:16,color:P.gray,align:'center'});
 L(283,268,310,268,{color:P.gold});L(310,268,310,104,{color:P.gold});A(310,104,330,104,{color:P.gold});

 T('Split each output row',327,43,228,25,{bold:true});
 M(335,78,4,8,13,{groups:['gold','gold','gold','gold','teal','teal','teal','teal']});
 S(334,77,54,54,'none',P.gold,1.5);S(387,77,54,54,'none',P.teal,1.5);
 T('W′',451,87,35,30,{size:25,italic:true});
 A(362,135,362,167,{color:P.gold});A(414,135,414,167,{color:P.teal});
 M(335,173,1,4,13,{palette:'gold'});M(405,173,1,4,13,{palette:'teal'});
 T('weight vectors wⱼ′',329,198,154,26,{size:17,italic:true});
 M(516,83,4,4,13,{palette:'blue',pattern:(r,c)=>r===c?4:0});T('M',530,49,44,28,{size:25,italic:true});
 T('Action sensitivity',484,139,110,43,{size:16,align:'center',color:P.blue});
 A(471,189,507,226,{color:P.teal});A(537,180,537,222,{color:P.blue});
 T('argmin dₘ(wⱼ′, c)',335,232,238,27,{size:20,color:P.teal,italic:true,align:'center'});
 M(357,277,3,4,17,{palette:'blue'});S(357,294,68,17,'none',P.teal,2.1);
 T('C',328,290,28,26,{size:25,italic:true});
 for(const [i,z] of [2,1,3].entries()){S(451,277+i*17,27,17,i===1?P.tealLight:'#F5F7FA',P.line,.65);T(String(z),451,277+i*17,27,17,{size:15,align:'center',middle:true,color:i===1?P.teal:P.ink});}
 T('z',456,254,31,24,{size:22,italic:true});T('codewords',350,337,88,24,{size:16,color:P.gray});T('indices',445,337,83,24,{size:16,color:P.gray});
 A(515,336,515,278,{color:P.teal,dash:true});T('Calibration',492,352,102,23,{size:16,color:P.teal});
 L(482,305,628,305,{color:P.teal});L(628,305,628,114,{color:P.teal});A(628,114,644,114,{color:P.teal});

 T('C, z reconstruct Ŵ′',639,46,227,28,{size:20,bold:true});
 M(649,97,3,6,12,{palette:'gold'});M(779,97,3,6,12,{palette:'blue'});
 T('Ŵ′',671,138,58,27,{size:24,italic:true});T('X′',799,138,40,27,{size:24,italic:true});
 T('BF16 first',774,75,96,23,{size:16,color:P.blue});
 A(685,166,716,206,{color:P.navy,width:1.4});A(814,166,785,206,{color:P.navy,width:1.4});
 for(let k=0;k<3;k++){S(676,211+k*23,146,18,k===0?P.tealLight:P.blueLight,k===0?P.teal:P.blue,.9);T(k===0?'VQ linear':'Remaining layers',676,211+k*23,146,18,{size:14.8,align:'center',middle:true});}
 A(749,279,749,303,{color:P.navy,width:1.6});M(707,315,3,7,12,{palette:'blue'});
 T('Final action chunk',648,354,221,23,{size:17,bold:true,align:'center'});
}

export function geometryA(s){
 const g=canvas(s,48,88),{S,T,L,A,D,E,M,panel,P}=g;
 panel('(a) Error direction',0,340);panel('(b) Distortion',365,248);panel('(c) Codebook metric',649,233);
 L(347,0,347,373,{color:P.line,dash:true});L(632,0,632,373,{color:P.line,dash:true});
 const cx=148,cy=205,u=147;
 // Contours are exact level sets of M=diag(9,1), not experimental curves.
 for(const [lev,col] of [[.81,'#C7D8EA'],[.36,'#91B9D2'],[.09,P.teal]])E(cx,cy,Math.sqrt(lev)/3*u,Math.sqrt(lev)*u,{stroke:col,width:1.1});
 E(cx,cy,.3*u,.3*u,{stroke:P.gold,dash:true});
 A(29,cy,313,cy,{color:P.gray,width:.8});A(cx,340,cx,59,{color:P.gray,width:.8});
 T('Sensitive direction',167,294,158,37,{size:16,color:P.gray});T('Less sensitive',9,43,118,31,{size:16,color:P.gray});
 D(cx,cy,P.navy);D(cx+.3*u,cy,P.gold);D(cx,cy-.6*u,P.teal);
 A(cx+5,cy,cx+.3*u-5,cy,{color:P.gold,width:2.1});A(cx,cy-6,cx,cy-.6*u+7,{color:P.teal,width:2.1});
 T('w = (0,0)',26,222,125,24,{size:19,italic:true});T('c₁ = (0.3,0)',204,215,132,26,{size:19,color:P.gold,italic:true});T('c₂ = (0,0.6)',167,105,151,25,{size:19,color:P.teal,italic:true});
 T('Gold: L₂ nearest\nTeal: action-weighted nearest',9,345,326,40,{size:16.5,color:P.gray});

 T('Squared distance',366,47,225,27,{size:20,bold:true});
 T('L₂',369,88,42,26,{size:22,color:P.gold});T('Action',438,88,106,26,{size:22,color:P.teal});
 T('c₁',367,128,38,24,{size:22,italic:true});S(404,136,.09*183,13,P.gold);T('0.09',433,124,58,26,{size:19,color:P.gold});
 S(404,168,.81*183,13,P.teal);T('0.81',557,157,60,26,{size:19,color:P.teal});
 T('c₂',367,219,38,24,{size:22,italic:true});S(404,228,.36*183,13,P.gold);T('0.36',478,216,60,26,{size:19,color:P.gold});
 S(404,260,.36*183,13,P.teal);T('0.36',478,248,60,26,{size:19,color:P.teal});
 T('L₂ selects c₁\nAction metric selects c₂',368,312,239,64,{size:20,bold:true});
 T('M = diag(9,1)',657,46,219,28,{size:22,italic:true});
 M(679,87,2,2,33,{palette:'blue',pattern:(r,c)=>r===c?(r===0?4:1):0});
 for(const [r,row] of [[9,0],[0,1]].entries())for(const [c,v] of row.entries())T(String(v),679+c*33,87+r*33,33,33,{size:23,align:'center',middle:true,color:r===0&&c===0?'#FFFFFF':P.navy});
 T('dₘ(w,c) =\n(w−c)ᵀM(w−c)',654,171,223,55,{size:21,italic:true});
 // Schematic anisotropic codebook spacing, 3:1 vertical/horizontal ratio.
 for(let r=0;r<3;r++)for(let c=0;c<6;c++)S(672+c*25,251+r*50,7,7,r===1&&c===3?P.teal:P.blueLight,P.blue,.8);
 E(750,304,17,51,{stroke:P.teal,fill:P.tealLight});D(750,304,P.teal,3.5);
 T('Illustrative directional resolution',652,370,225,20,{size:14.5,color:P.gray});
}

export function methodB(s){
 const g=canvas(s,48,88),{S,T,L,A,M,E,D,panel,P}=g;
 panel('(a) Equivalent conditioning',0,326);panel('(b) Action-sensitive VQ',336,323);panel('(c) Inference',688,188);
 L(321,0,321,374,{color:P.line,dash:true});L(674,0,674,374,{color:P.line,dash:true});
 T('Layer activations X',4,43,177,24,{size:18,bold:true});T('X′',262,45,45,24,{size:22,italic:true});
 M(10,79,4,6,12,{pattern:(r,c)=>c===4?4:(r+c)%3});M(236,79,4,6,12,{pattern:(r,c)=>(r+2*c)%3});
 S(139,85,83,48,P.blueLight,P.blue,1);T('D⁻¹H',139,88,83,33,{size:24,align:'center',middle:true,color:P.blue});
 A(89,106,133,106,{color:P.blue,width:1.7});A(222,106,232,106,{color:P.blue,width:1.7});
 T('Scale + Hadamard',110,140,194,24,{size:17,color:P.blue});
 T('Layer weights W',4,184,177,24,{size:18,bold:true});T('W′',262,184,45,24,{size:22,italic:true});
 M(10,220,4,6,12,{palette:'gold'});M(236,220,4,6,12,{palette:'gold',pattern:(r,c)=>(r+c)%5});
 S(139,226,83,48,P.goldLight,P.gold,1);T('DH',139,229,83,33,{size:24,align:'center',middle:true,color:P.gold});
 A(89,247,133,247,{color:P.gold,width:1.7});A(222,247,232,247,{color:P.gold,width:1.7});
 S(3,307,307,60,'#F4F6FA');T('X′(W′)ᵀ = XWᵀ',10,314,294,27,{size:22,bold:true,align:'center'});T('D invertible, HHᵀ = I',10,342,294,25,{size:18,italic:true,align:'center',color:P.gray});
 L(308,244,327,244,{color:P.gold});L(327,244,327,90,{color:P.gold});A(327,90,339,90,{color:P.gold});

 T('One transformed output row',343,43,319,24,{size:18,bold:true});
 for(const [k,pal] of ['gold','red','teal'].entries())M(345+k*102,80,1,4,20,{palette:pal});
 T('block 1',357,107,75,24,{size:17,italic:true});T('block 2',459,107,75,24,{size:17,italic:true});T('block K',561,107,75,24,{size:17,italic:true});
 A(481,134,481,157,{color:P.navy,width:1.6});
 S(343,162,313,151,P.tealLight,P.teal,1,'roundRect',true);
 T('Sensitive codebook C + indices z',353,172,293,25,{size:19,bold:true,color:P.teal});
 M(355,218,1,4,16,{palette:'gold'});A(424,226,465,226,{color:P.teal,width:1.6});
 T('argmin',416,198,53,19,{size:15.5,italic:true,color:P.teal});
 M(474,209,4,4,16,{palette:'blue'});S(474,241,64,16,'none',P.teal,2.2);
 for(const [k,v] of [3,1,3,2].entries()){S(561,209+k*16,26,16,k===2?'#C6E1DB':'#FFFFFF',P.blue,.5);T(String(v),561,209+k*16,26,16,{size:13.5,align:'center',middle:true});}
 T('C',491,277,28,24,{size:20,italic:true});T('z',568,277,26,24,{size:20,italic:true});
 T('(w′ − C[z])ᵀ M (w′ − C[z])',352,292,293,23,{size:16.5,italic:true,align:'center'});
 A(477,357,477,316,{color:P.teal,dash:true,width:1.4});T('M from final-action sensitivity',351,330,298,25,{size:17.5,bold:true,color:P.teal,align:'center'});
 L(656,260,680,260,{color:P.teal});L(680,260,680,99,{color:P.teal});A(680,99,690,99,{color:P.teal});

 T('X′',798,45,36,25,{size:23,italic:true});T('Ŵ′',704,45,45,25,{size:23,italic:true});
 M(695,81,3,5,12,{palette:'gold'});M(797,81,3,5,12,{palette:'blue'});
 T('C,z decode',690,122,96,24,{size:16,color:P.gold});T('BF16 first',789,122,84,24,{size:16,color:P.blue});
 A(725,151,769,180,{color:P.navy});A(827,151,785,180,{color:P.navy});
 S(719,185,130,46,'#F2F4F9',P.navy,1.2);T('Linear',719,193,130,27,{size:23,bold:true,align:'center'});
 A(784,233,784,252,{color:P.navy,width:1.5});
 T('Remaining ActionDiT\n+ denoising',690,255,187,41,{size:17,align:'center'});
 A(784,298,784,312,{color:P.navy,width:1.5});M(742,318,3,7,12,{palette:'blue'});
 T('Final actions',712,358,154,24,{size:18,bold:true,align:'center'});
}

function assignmentPlot(g,x,weighted){
 const {S,T,L,A,D,E,P}=g,cx=x+145,cy=214,u=165;
 // Both plots use the same coordinates, and exact radius/axis ratio.
 E(cx,cy,weighted?.2*u:.3*u,weighted?.6*u:.3*u,{fill:weighted?P.tealLight:P.goldLight,stroke:weighted?P.teal:P.gold,dash:weighted});
 A(x+24,cy,x+282,cy,{color:P.gray,width:.8});A(cx,342,cx,62,{color:P.gray,width:.8});
 for(const v of [-.6,-.3,.3,.6]){L(cx+u*v,cy-3,cx+u*v,cy+3,{color:P.gray,width:.6});T(String(v),cx+u*v-16,cy+8,36,20,{size:13.5,align:'center',color:P.gray});}
 for(const v of [.3,.6]){L(cx-3,cy-u*v,cx+3,cy-u*v,{color:P.gray,width:.6});T(String(v),cx-35,cy-u*v-10,26,20,{size:13.5,color:P.gray});}
 D(cx,cy,P.navy);D(cx+.3*u,cy,P.gold);D(cx,cy-.6*u,P.teal);
 A(cx+5,cy,cx+.3*u-6,cy,{color:weighted?'#C3AD82':P.gold,width:weighted?.8:2});
 if(weighted)A(cx,cy-5,cx,cy-.6*u+6,{color:P.teal,width:2});
 T('c₁=(0.3,0)',cx+.3*u+12,cy-26,114,23,{size:16.5,color:P.gold,italic:true});
 T('c₂=(0,0.6)',cx+12,cy-.6*u-17,123,24,{size:16.5,color:P.teal,italic:true});
 T('w=(0,0)',cx-110,cy+37,106,23,{size:17.5,italic:true});
 T('Sensitive',x+187,281,103,22,{size:15,color:P.gray});T('Less sensitive',x+2,43,121,21,{size:15,color:P.gray});
 const fill=weighted?P.tealLight:P.goldLight,col=weighted?P.teal:P.gold;
 S(x+3,313,294,59,fill);T(weighted?'dₘ: c₁=0.81, c₂=0.36':'d₂: c₁=0.09, c₂=0.36',x+13,320,274,24,{size:19,italic:true,color:col});T(weighted?'Action metric selects c₂':'L₂ selects c₁',x+13,346,274,24,{size:19,bold:true,color:col});
}

export function geometryB(s){
 const g=canvas(s,48,88),{T,S,L,D,E,panel,P}=g;
 panel('(a) Euclidean assignment',0,303);panel('(b) Action-weighted VQ',319,313);panel('(c) Metric geometry',648,231);
 L(308,0,308,374,{color:P.line});L(637,0,637,374,{color:P.line});
 assignmentPlot(g,0,false);assignmentPlot(g,325,true);
 S(649,44,229,125,'#F4F6FA');T('dₘ(w,c) =',660,52,205,27,{size:24,italic:true});T('(w−c)ᵀM(w−c)',660,83,205,28,{size:23,italic:true});
 T('M =',660,126,55,29,{size:25,italic:true});
 T('9  0\n0  1',726,115,65,49,{size:23,align:'center'});L(720,116,720,166,{color:P.navy});L(717,116,723,116,{color:P.navy});L(717,166,723,166,{color:P.navy});L(798,116,798,166,{color:P.navy});L(795,116,801,116,{color:P.navy});L(795,166,801,166,{color:P.navy});
 T('Equal action-error contours',651,186,225,41,{size:19,bold:true});
 for(let r=0;r<3;r++)for(let c=0;c<6;c++)S(663+c*33,230+r*47,7,7,P.blueLight,P.blue,.7);
 E(765,280,15,45,{fill:P.tealLight,stroke:P.teal,dash:true});D(765,280,P.teal,4);
 T('Vertical / horizontal\nradius ratio = 3',653,334,223,42,{size:18,color:P.gray});
}
