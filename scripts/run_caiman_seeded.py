import os
import sys

# sys.path.append('G:/My Drive/Code/holofun')
sys.path.append('c:/users/will/code/analysis')

from holofun.tiffs import SItiff

data_root = ''
mouse = ''
date = ''
epoch = ''

pth = os.path.join(data_root, mouse, date, epoch)


def make_ain_from_tiff(tiff, plane, channel=1):
    tiff = SItiff('tiff')