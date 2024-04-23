import matplotlib.pyplot as plt
from holofun.tiffs import SItiff
from holofun.simple_guis import openfilegui

from pathlib import Path
import numpy as np

do_rgb = False
cmap = 'cividis'
ch = 1 # 0 for green, 1 for red

root_dir = '/mnt/hdd/data2/experiments'

file = openfilegui(root_dir)
print('loading file...')
tiff = SItiff(file)

all_pts = []
for p in tiff.zs:
    img_data = tiff.mean_img(p, ch, as_rgb=do_rgb)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(img_data, cmap=cmap)
    ax.set_title(f'Click the cells. (z = {p})')
    im_min = 0
    im_max = img_data.max()
    pts = fig.ginput(n=-1, timeout=-1)
    pts = [(*pt, p) for pt in pts]
    all_pts.append(pts)

fpath = Path(file).parent
spath = fpath.parent/'clicked_cell_locs.npy'
all_pts = [item for sublist in all_pts for item in sublist]
np.save(spath, all_pts)
print(f'Saved to: {spath}')