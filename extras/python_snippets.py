# remove flyback artifacts
# paste in suite2p.io.tiff.py, approx line 175, right after the tiff gets read
if 'remove_artifacts' in ops.keys():
    im = im[:,:,slice(*ops['remove_artifacts'])]


# in views.py, around line 50, play with this to adjust the image view
elif k==1:
    mimg = parent.ops['meanImg']
    mimg1 = np.percentile(mimg,1)
    mimg99 = np.percentile(mimg,99)
    mimg     = (mimg - mimg1) / (mimg99 - mimg1)
    # THESE LINES BELOW...
    mimg = np.maximum(0,np.minimum(1.5,mimg)) 
    mimg = (mimg - mimg.min()) / (mimg.max() - mimg.min())