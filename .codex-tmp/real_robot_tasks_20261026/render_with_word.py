from pathlib import Path
import subprocess, sys
import win32com.client
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parent
source=Path(r'D:\Downloads\Final Year Project\output\documents\2026-10-26_FastWAM_RealRobot_Task_Recommendations.docx')
render=root/'render'
render.mkdir(exist_ok=True)
pdf=render/'review.pdf'
word=win32com.client.DispatchEx('Word.Application')
word.Visible=False
word.DisplayAlerts=0
document=None
try:
    document=word.Documents.Open(str(source),ReadOnly=True,AddToRecentFiles=False)
    document.Fields.Update()
    document.ExportAsFixedFormat(str(pdf),17)
finally:
    if document is not None: document.Close(False)
    word.Quit()
reader=PdfReader(pdf)
print('PAGES',len(reader.pages))
for i,page in enumerate(reader.pages,1):
    text=page.extract_text()
    print(i,len(text),repr(text[:95]),repr(text[-100:]))
poppler=Path(r'C:\Users\Raymond\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe')
subprocess.run([str(poppler),'-r','120','-png',str(pdf),str(render/'page')],check=True)
print('PNG_COUNT',len(list(render.glob('page-*.png'))))
