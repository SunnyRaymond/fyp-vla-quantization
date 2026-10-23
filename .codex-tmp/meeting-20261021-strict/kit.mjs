// Source coordinates match the 1672x941 ImageGen references; output uses native slide objects.
export function createKit(slide,{left=27,top=0,width=906}={}) {
 const k=width/1672, X=x=>left+x*k, Y=y=>top+y*k, S=x=>x*k;
 let count=0;const records=[];
 const pos=(x,y,w,h)=>({left:X(x),top:Y(y),width:S(w),height:S(h)});
 function text(str,x,y,w,h,o={}){
  const sh=slide.shapes.add({name:o.name||String(str).slice(0,75),geometry:'textbox',position:{...pos(x,y,w,h),...(o.rotation?{rotation:o.rotation}:{})},fill:o.fill??'none',line:{fill:o.border??'none',width:S(o.borderWidth??0)}});
  sh.text=String(str).split('\n').map(v=>({runs:[v],spaceBefore:0,spaceAfter:0}));
  sh.text.style={typeface:o.font??'Arial Narrow',fontSize:S(o.size??20),bold:o.bold??false,italic:o.italic??false,color:o.color??'#061A3C',alignment:o.align??'left',verticalAlignment:o.middle?'middle':'top',autoFit:'shrinkText',wrap:o.wrap??'none',lineSpacing:o.lineSpacing??1.0,insets:0};
  records.push({type:'text',str,x,y,w,h,o});count++;return sh;
 }
 function rect(x,y,w,h,o={}){
  const sh=slide.shapes.add({name:o.name??'diagram block',geometry:o.geometry??'rect',position:pos(x,y,w,h),fill:o.fill??'#FFFFFF',line:{fill:o.line??'#1A375B',width:S(o.width??1),style:o.dash?'dashed':'solid'},...(o.radius?{borderRadius:S(o.radius)}:{})});
  records.push({type:'rect',x,y,w,h,o});count++;return sh;
 }
 function box(str,x,y,w,h,o={}){
  const sh=rect(x,y,w,h,o); if(str){const pad=o.pad??5;text(str,x+pad,y+pad,w-pad*2,h-pad*2,{...o,align:o.align??'center',middle:o.middle??true});}return sh;
 }
 function line(x1,y1,x2,y2,o={}){
  const sh=slide.shapes.add({name:o.name??'native PowerPoint connector',geometry:'connector',kind:o.kind??'straight',position:{left:X(Math.min(x1,x2)),top:Y(Math.min(y1,y2)),width:S(Math.abs(x2-x1)),height:S(Math.abs(y2-y1)),horizontalFlip:x2<x1,verticalFlip:y2<y1},line:{fill:o.color??'#071B3C',width:S(o.width??1.7),style:o.dash?'dashed':'solid'},...(o.start?{head:{type:o.start===true?'triangle':o.start,width:o.endSize??'sm',length:o.endSize??'sm'}}:{}),...(o.end?{tail:{type:o.end===true?'triangle':o.end,width:o.endSize??'sm',length:o.endSize??'sm'}}:{})});
  records.push({type:'line',x1,y1,x2,y2,o});count++;return sh;
 }
 function arrow(x1,y1,x2,y2,o={}){return line(x1,y1,x2,y2,{end:true,...o});}
 function poly(points,o={}){return points.slice(1).map((p,i)=>line(points[i][0],points[i][1],p[0],p[1],{...o,end:i===points.length-2?(o.end??true):false}));}
 function nativeArrow(x,y,w,h,o={}){return rect(x,y,w,h,{geometry:o.geometry??'rightArrow',fill:o.fill??'#07366E',line:o.line??'none',width:0,...o});}
 function custom(points,x,y,w,h,o={}){
  const sh=slide.shapes.add({geometry:'custom',name:o.name??'native vector shape',position:pos(x,y,w,h),fill:o.fill??'none',line:{fill:o.line??'#0A396A',width:S(o.width??1.2)},customPaths:[{width:w,height:h,commands:[{moveTo:{x:points[0][0],y:points[0][1]}},...points.slice(1).map(p=>({lineTo:{x:p[0],y:p[1]}})),...(o.close===false?[]:[{close:{}}])]}]});count++;return sh;
 }
 function fraction(num,den,x,y,w,h,o={}){const size=o.size??25;text(num,x,y,w,h*.42,{...o,size,align:'center',font:o.font??'Times New Roman',italic:true});line(x+3,y+h*.49,x+w-3,y+h*.49,{color:o.color??'#061A3C',width:1.1});text(den,x,y+h*.52,w,h*.42,{...o,size,align:'center',font:o.font??'Times New Roman',italic:true});}
 function math(str,x,y,w,h,o={}){return text(str,x,y,w,h,{font:'Times New Roman',italic:true,size:26,...o});}
 function grid(x,y,rows,cols,w,h,o={}){const gap=o.gap??0;for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const fill=typeof o.fill==='function'?o.fill(r,c):o.fill;rect(x+c*w/cols,y+r*h/rows,w/cols-gap,h/rows-gap,{...o,fill});}}
 return {slide,k,text,rect,box,line,arrow,poly,nativeArrow,custom,fraction,math,grid,records,get count(){return count;}};
}
