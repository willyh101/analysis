import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

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
    plt.clf()
    img = tiff.mean_img(p, ch, as_rgb=do_rgb)
    fig, ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(bottom=0.25)
    im = ax.imshow(img, cmap=cmap)
    ax.set_title(f'Click the cells. (z = {p})')

    im_min = 0
    im_max = img.max()

    slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Threshold", im_min, im_max, valinit=(im_min, im_max))

    def update(val):
        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    pts = plt.ginput(n=-1, timeout=-1)
    pts = [(*pt, p) for pt in pts]
    
fpath = Path(file).parent
spath = fpath.parent/'clicked_cell_locs.npy'
all_pts = [item for sublist in all_pts for item in sublist]
np.save(spath, all_pts)
print(f'Saved to: {spath}')
