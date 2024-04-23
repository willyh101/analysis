#!/usr/bin/env python

# assumes to be running in the 'analysis' conda environment

from pathlib import Path
from holofun.tiffs import SItiff
from holofun.utils import list_dirs


mouse = 'w71_1'
date = '20240408'

tiff_root = '/mnt/hdd/data2/experiments'
pth = Path(tiff_root, mouse, date)
folders = list_dirs(pth)

for folder in folders:
    print(folder)

    scan_files = Path(tiff_root, mouse, date, folder).glob('*.tif*')

    num_channels = []
    num_planes = []
    num_frames = []
    consistent_frames = []

    for file in scan_files:
        tiff = SItiff(file)
        num_channels.append(tiff.nchannels)
        num_planes.append(tiff.nplanes)

        # extract each single plane and channel=0
        # and get the number of frames
        nf = []
        for z in range(tiff.nplanes):
            data = tiff.extract(z, 0)
            nframes_z = data.shape[0]
            nf.append(nframes_z)
        consistent_frames.append(all([n == nf[0] for n in nf]))
        num_frames.append(tuple(nf))

    print(f'Consistent channels? {all([n == num_channels[0] for n in num_channels])}')
    print(f'Consistent planes? {all([n == num_planes[0] for n in num_planes])}')

    # accept_frame_range = range(num_frames[0][0]-5, num_frames[0][0]+5)
    # consistent_tiffs = [n in accept_frame_range for n in num_frames[0]]
    # is_consistent_frames = all(consistent_tiffs)
    # find the index of the files that have inconsistent frames
    # print(f'Consistent frames? {is_consistent_frames}')

    print('Conistent frames? ', all(consistent_frames))

    print('\n -----')


