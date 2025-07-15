import re
from typing import Callable
from IPython.display import Markdown, Image as IPyImage, display

def replace_image_links(md:str, repl_fn:Callable[[str],str]) -> str:
    # repl_fn รับ filename -> string ที่จะใส่แทน
    pattern = re.compile(r'!\[img-\d+\.jpeg\]\((img-\d+\.jpeg)\)')
    return pattern.sub(lambda m: repl_fn(m.group(1)), md)

def render_with_s3(md, storage):
    pattern = re.compile(r'!\[.*?\]\((.*?)\)')
    pos = 0
    for m in pattern.finditer(md):
        if m.start() > pos:
            display(Markdown(md[pos:m.start()]))
        fname = m.group(1).split('/')[-1]
        image_id = fname.split('.',1)[0]
        try:
            display(IPyImage(data=storage.get_binary(image_id)))
        except Exception:
            display(Markdown(m.group(0)))
        pos = m.end()
    if pos < len(md):
        display(Markdown(md[pos:]))
