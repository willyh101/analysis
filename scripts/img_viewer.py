import argparse
import pprint

import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

from holofun.tiffs import SItiff
from holofun.simple_guis import openfilegui

print_info = True
do_rgb = False
cmap = 'cividis'

# ROOT_DIR = '/mnt/data2/experiments/w68_1/20231108'
# ROOT_DIR = '/media/will/DATA 2/experiments/w68_1/20231108'
# ROOT_DIR = '/mnt/hdd/data2/experiments'
ROOT_DIR = '/mnt/servers/frankenshare/Will/ScanImage Data'
Z_PLANE = 1
CH = 1 # 0 for green, 1 for red

# parser = argparse.ArgumentParser(description='View a tiff image.')
# parser.add_argument('')

# load image
file = openfilegui(ROOT_DIR)
print('loading file...')
tiff = SItiff(file)
img = tiff.mean_img(Z_PLANE, CH, as_rgb=do_rgb)

print('\n******SI METADATA*******\n')
pprint.pprint(tiff.metadata)

fig, ax = plt.subplots(figsize=(5,5))
fig.subplots_adjust(bottom=0.25)

im = ax.imshow(img, cmap=cmap)

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
plt.show()
print('goodbye!')