"""Regenerate editable SVG/draw.io diagrams. Standard library only."""
from pathlib import Path
from html import escape
import xml.etree.ElementTree as ET
OUT = Path(__file__).resolve().parent

def diagram(name, width, height, nodes, edges, caption):
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',f'<title id="title">{escape(caption)}</title><desc id="desc">节点和连线均可编辑</desc>', '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,1 L8,5 L0,9" fill="none" stroke="#64748b" stroke-width="1.5"/></marker></defs>', '<rect width="100%" height="100%" fill="#ffffff"/>']
    mx = ET.Element('mxfile', host='app.diagrams.net'); page = ET.SubElement(mx,'diagram',name=caption,id=name)
    model = ET.SubElement(page,'mxGraphModel',dx=str(width),dy=str(height),grid='1',gridSize='10',page='1',pageWidth=str(width),pageHeight=str(height))
    root = ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    lookup = {n[0]:n for n in nodes}
    for index,(a,b,points,dashed) in enumerate(edges):
        path = 'M'+' L'.join(f'{x},{y}' for x,y in points)
        dash = 'stroke-dasharray="6 5"' if dashed else ''
        svg.append(f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="2" {dash} marker-end="url(#arrow)"/>')
        cell = ET.SubElement(root,'mxCell',id=f'edge{index}',parent='1',edge='1',source=a,target=b,style='edgeStyle=orthogonalEdgeStyle;rounded=1;endArrow=block;strokeColor=#64748b;'+('dashed=1;' if dashed else ''))
        geo=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        if len(points)>2:
            arr=ET.SubElement(geo,'Array',**{'as':'points'})
            for x,y in points[1:-1]:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
    for ident,x,y,w,h,lines,kind in nodes:
        fill,stroke = {'core':('#edf6f4','#0f766e'),'ai':('#edf3fb','#416da5'),'plan':('#fff8eb','#b8832f'),'plain':('#f8fafc','#cbd5e1')}[kind]
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        center=y+h/2-(len(lines)-1)*14
        for i,line in enumerate(lines):
            svg.append(f'<text x="{x+w/2}" y="{center+i*28}" text-anchor="middle" dominant-baseline="middle" font-family="PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif" font-size="{20 if i==0 else 17}" font-weight="{600 if i==0 else 400}" fill="#172b40">{escape(line)}</text>')
        cell = ET.SubElement(root,'mxCell',id=ident,parent='1',vertex='1',value='\n'.join(lines),style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};fontColor=#172b40;fontSize=18;spacing=10;')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
    svg.append('</svg>');(OUT/(name+'.svg')).write_text('\n'.join(svg),encoding='utf-8')
    ET.indent(mx);ET.ElementTree(mx).write(OUT/(name+'.drawio'),encoding='utf-8',xml_declaration=True)

diagram('workflow',1200,570,[
 ('data',20,200,175,110,['数据准备','确认病例与序列'],'plain'),
 ('batch',260,30,240,100,['批量草稿','适用模型自动预测'],'ai'),
 ('prompt',260,205,240,100,['交互标注','点／框／涂鸦引导'],'ai'),
 ('manual',260,380,240,100,['手工标注','Mimics 原生编辑'],'plain'),
 ('review',575,200,240,110,['人工检查与精修','三视图核对与保存'],'core'),
 ('output',885,200,290,110,['确认后导出与复用','标签标准及质量检查'],'core'),
 ('train',860,410,300,95,['模型适配与管理','已有标注 → 训练 → 预测'],'ai'),
 ('state',575,425,230,65,['完善复核与训练准入'],'plan')],
 [('data','batch',[(195,235),(225,235),(225,80),(260,80)],False),('data','prompt',[(195,255),(260,255)],False),('data','manual',[(195,275),(225,275),(225,430),(260,430)],False),('batch','review',[(500,80),(535,80),(535,230),(575,230)],False),('prompt','review',[(500,255),(575,255)],False),('manual','review',[(500,430),(535,430),(535,280),(575,280)],False),('review','output',[(815,255),(885,255)],False),('output','train',[(1030,310),(1030,410)],False),('train','batch',[(1160,455),(1185,455),(1185,10),(380,10),(380,30)],False),('state','review',[(690,425),(690,310)],True)],'三条标注路径与模型迭代')

diagram('architecture',1200,520,[
 ('host',25,100,270,185,['Mimics 工作台','影像查看 · Mask 精修','人工确认 · 工程保存'],'core'),
 ('bridge',360,120,255,145,['脚本桥接','身份与空间匹配','任务控制 · 结果写入'],'plain'),
 ('external',685,60,475,115,['外部计算进程','数据转换 · AI 推理 · 模型训练'],'ai'),
 ('local',685,250,215,100,['本地 GPU','适合交互与单例'],'ai'),
 ('remote',940,250,220,100,['远程 GPU','按功能支持训练／预测'],'ai'),
 ('support',360,395,800,80,['共同支撑','状态反馈 · 环境诊断 · 模型管理 · 结果与冲突记录'],'plain')],
 [('host','bridge',[(295,170),(360,170)],False),('bridge','host',[(360,230),(295,230)],False),('bridge','external',[(615,160),(650,160),(650,115),(685,115)],False),('external','bridge',[(685,145),(655,145),(655,225),(615,225)],False),('external','local',[(790,175),(790,250)],False),('external','remote',[(1050,175),(1050,250)],False)],'Mimics 与外部计算的分工')
