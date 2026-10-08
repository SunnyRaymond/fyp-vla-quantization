from pathlib import Path
from PIL import Image
folder = Path(__file__).parent / 'papers'
# Extract original PDF pixels. Captions remain available in the source images.
for name, box in [
    ('quantwams-fig2-method', (0, 0, 2289, 970)),
    ('q-wam-fig2-method', (0, 0, 2289, 1200)),
    ('steerquant-fig2-method', (0, 100, 2289, 1335)),
]:
    im = Image.open(folder / (name + '.png'))
    im.crop(box).save(folder / (name + '-main.png'))
# A magnified W4A4 view preserves headers and original row pixels.
im = Image.open(folder / 'steerquant-table1-fastwamjoint-panel.png')
zoom = Image.new('RGB', (im.width, 370), 'white')
zoom.paste(im.crop((0, 0, im.width, 151)), (0, 0))
zoom.paste(im.crop((0, 438, im.width, 657)), (0, 151))
zoom.save(folder / 'steerquant-table1-w4a4-zoom.png')
