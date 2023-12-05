## TODO:
### Loading:
- [ ] get offsets (from motion correction)
- [ ] motion correct?
- [ ] he has something with DS but I don't know what that is? I think it has to do with offsets
- [ ] something weird with cellvce, theROIs and a for loop that gets xs, ys, and makes theROIs with poly2mask
- [ ] makes a theROIimg with the masks
- [ ] get stim information with stimReader


### Tests:
- [ ] make tests for main code


### Other
- [x] fix setup.py scrips, maybe add bin to path?
    - I have currently removed scripts from setup.py because they cache into miniforge instead of being added to the path, so this makes them not editable
    - [x] add bin to path programmatically

- [ ] fix PyQT GUI interfaces
