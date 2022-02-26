## TODO:
### Loading:
* get frame numbers via the Python s2p output
* get metadata and framerate
* load Fcell, FcellNeu, sp
* get offsets (from motion correction)
* motion correct?
* he has something with DS but I don't know what that is? I think it has to do with offsets
* split up Fcell by epochs
* get ROI medians
* something weird with cellvce, theROIs and a for loop that gets xs, ys, and makes theROIs with poly2mask
* makes a theROIimg with the masks
* zscore of otherwise
* get stim information with stimReader
* get run speed with runReader
* exclude cells in artifact
* something about 'limit T to relevant frames for use with continuous imaging'