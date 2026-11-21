"""Build the offline UI font subset from Google's OFL NotoSansSC[wght].ttf.

Usage: python tool/subset_stair_ui_font.py /path/to/NotoSansSC.ttf
Requires fonttools; preserve the upstream OFL.txt next to the output font.
"""
import argparse
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools import subset

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('font', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / 'app/frontend'
text = ''.join(p.read_text() for p in (root/'lib').rglob('*.dart'))
font = instantiateVariableFont(TTFont(args.font), {'wght': 400}, inplace=True)
for record in font['name'].names:
    names = {1: 'TinyNav UI Sans', 3: 'TinyNavUISans-Regular', 4: 'TinyNav UI Sans Regular',
             6: 'TinyNavUISans-Regular', 16: 'TinyNav UI Sans', 17: 'Regular'}
    if record.nameID in names:
        record.string = names[record.nameID].encode(record.getEncoding())
sub = subset.Subsetter()
sub.populate(text=''.join(c for c in text if ord(c) > 127))
sub.subset(font)
out = root/'assets/fonts/NotoSansSC/TinyNavUI.ttf'
out.parent.mkdir(parents=True, exist_ok=True)
font.save(out)
print(out)
