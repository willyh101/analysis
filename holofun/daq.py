from .simple_guis import openfilegui
import numpy as np
import h5py

# going to start by writing functions but eventially this
# should be folded into a DaqFile class
# a lot of this can be taken from analysis_main.py


class SetupDaqFile:
    def __init__(self, path, epoch, fr, debug=False):
        self.path = path # path to .mat file
        self.epoch = epoch - 1 # matlab -> python indexing fix
        self.fr = fr # frame rate
        
        self.rate = 20000
        self.pt_flip_ch = 5
        self.pt_cond_ch = 3
        self.run_ch = 0
        self.wheel_circum = 47.75
        
        # experiment info
        self.date = decode(path, 'ExpStruct/date')
        self.mouse = decode(path, 'ExpStruct/Expt_Params/ExpType')
        self.genotype = decode(path, 'ExpStruct/Expt_Params/genotype')
        self.virus = decode(path, 'ExpStruct/Expt_Params/virus')
        self.imaging_params = decode(path, 'ExpStruct/Expt_Params/slice')
        self.pmts = decode(path, 'ExpStruct/Expt_Params/PMTs')
        
        if not debug:  
            # vis and daq things
            self.sweeps = self.get_sweeps()
            self.vis_stims = self.get_vis_stim()
            self.vis_times = self.get_vis_times()
            self.running = self.get_running()
            self.mean_running = self.running.mean(axis=1)
            
            try:
                self.out_id = self.get_out_id()
            except KeyError:
                print('Has no outID!')
            
            # holo stuff
            self.hrnum = self.get_hrnum()
            try:
                self.targets = self.get_targets()
                self.rois = self.get_rois()
                self.stim_times = self.get_stim_times()
            except KeyError: # there can sometimes be an off-by-one due to daq weirdness
                self.hrnum -= 1
                self.targets = self.get_targets()
                self.rois = self.get_rois()
                self.stim_times = self.get_stim_times()
        
    def get_hrnum(self):
        with h5py.File(self.path, 'r') as f:
            hr_nums =  f['ExpStruct/Holo/Sweeps_holoRequestNumber'][:].squeeze().astype('uint8')[self.sweeps]
            try:
                assert np.all(hr_nums == hr_nums[0]), 'All holoRequests must be equal within an epoch. Check your epochs/sweep numbers.'
                hrnum = hr_nums[0] - 1 # just take the first hr number from the list
            except IndexError:
                raise IndexError('This holoRequest does not exist.')
        return hrnum
            
    def get_sweeps(self):
        """this is set up to have an extra trial for slicing"""
        with h5py.File(self.path, 'r') as f:
            try:
                enter_sweep = int(np.array(f[f['ExpStruct/EpochEnterSweep'][()][self.epoch,0]]).squeeze())-1
                try:
                    exit_sweep = int(np.array(f[f['ExpStruct/EpochEnterSweep'][()][self.epoch+1,0]]).squeeze())-1
                except IndexError:
                    exit_sweep = int(f['ExpStruct/sweep_counter'][()].squeeze())-1
                sweeps = np.arange(enter_sweep,exit_sweep,1)
            except IndexError:
                raise IndexError('Epoch does not exist.')
        return sweeps
    
    def get_vis_stim(self):
        with h5py.File(self.path, 'r') as f:
            vis_stims = []
            for s in self.sweeps:
                arr = f[f['ExpStruct/digitalSweeps'][s][0]][self.pt_cond_ch,:]
                vis_stims.append(count_daq_pulses(arr))
        return np.array(vis_stims)
    
    def get_vis_times(self):
        with h5py.File(self.path, 'r') as f:
            start = []
            stop = []
            for s in self.sweeps:
                swp = f[f['ExpStruct/digitalSweeps'][s][0]][self.pt_flip_ch,:]
                ts = np.where(np.diff(swp))[0]

                # append vis stim times
                try:
                    # this will fail if we missed a vis stim entirely (but not for null conditions)
                    start.append(ts[0]/self.rate)
                    stop.append(ts[1]/self.rate)

                except:
                    start.append(np.nan)
                    stop.append(np.nan)
                    
            vis_times = np.array([start, stop])
                    
        return np.array(vis_times)
    
    def get_running(self):
        """Get the mean running speed per trial."""
        with h5py.File(self.path, 'r') as f:
            trial_run_speed = []
            for s in self.sweeps:
                # get rotor ticks
                rotary_sweep = f[f['ExpStruct/digitalSweeps'][s][0]][self.run_ch,:]
                run_ticks = np.append(np.diff(rotary_sweep>0), 0)
                
                # calculate the number of daq pts per frame
                binwidth = 1 / self.fr  # width in time
                binpts = round(self.rate * binwidth) # width in pts
                
                # get the number of rotor ticks per bin and calc speed
                run_frame_bins = np.add.reduceat(run_ticks, np.arange(0,len(run_ticks), binpts))
                trial_speed = run_frame_bins / 360 * self.wheel_circum / binwidth
                trial_run_speed.append(trial_speed)   
        return np.array(trial_run_speed)
    
    def get_out_id(self):
        """This should ID stims trialwise."""
        with h5py.File(self.path, 'r') as f:
            all_outs = f['ExpStruct/outID'][:].squeeze().astype(np.int)
            outs_this_epoch = all_outs[self.sweeps]
        return outs_this_epoch
    
    def get_rois(self):
        with h5py.File(self.path, 'r') as f:
            ref = f['ExpStruct/Holo/holoRequests'][self.hrnum,0]
            rois_ref = f[ref]['rois'][:].squeeze()
            rois = [f[r][:].reshape((-1)).astype(np.int) for r in rois_ref]
        return rois
            
    def get_targets(self):
        with h5py.File(self.path, 'r') as f:
            ref = f['ExpStruct/Holo/holoRequests'][self.hrnum,0]
            targets = f[ref]['targets'][:].T.astype(np.int)
            targets[targets < 0] = 0
            targets[:] = targets[:, [1,0,2]] # puts them in x,y,z
        return targets
    
    def get_roi_weights(self):
        with h5py.File(self.path, 'r') as f:
            ref = f['ExpStruct/Holo/holoRequests'][self.hrnum,0]
            weights = f[ref]['roiWeights'][:].squeeze()
        return weights
    
    def get_stim_times(self):
        with h5py.File(self.path, 'r') as f:
            ref = f['ExpStruct/Holo/holoRequests'][self.hrnum,0]
            times = f[ref]['bigListOfFirstStimTimes'][0,:]
            times = times[~np.isnan(times)]
            return times
        
    def get_stims_legacy(self):
        print('Working on finding stimIDs using old approach...')
        with h5py.File(self.path, 'r') as f:
            all_output_names = []
            for outname_ref in f['ExpStruct/output_names'][:].squeeze():
                out_name_array = f[outname_ref][:].squeeze()
                out_name = u''.join(chr(c) for c in out_name_array)
                all_output_names.append(out_name)
                
            all_output_names[0] = 'none'
            
            # get the stim tags and calculate the unique stims
            stim_tags = f['ExpStruct/stim_tag'][:].squeeze()
            unique_stims = np.unique(stim_tags[self.sweeps]).astype('int8')
            output_patterns = f['ExpStruct/output_patterns'][:].squeeze()
            output_num = []
            output_name = []
            output_string = []
            
            # match those unique stims
            for s in unique_stims:
                this_output = f[f[f['ExpStruct/stimlog'][:].squeeze()[int(s-1)]][:][0,0]][:]
                is_found = False
                c = -1

                while not is_found:
                    c += 1
                    try:
                        test_output = f[output_patterns[c]]
                    except IndexError:
                        print('stims not unfucked!')
                        print(f'currently on c={c}')
                        print(f'trying to find unique_stim: {s}')
                        raise ValueError('Output not found')

                    if test_output.size == this_output.size:
                        is_found = np.all(this_output[:]==test_output[:])
                    if c > len(all_output_names):
                        is_found == True
                        c=1
                        raise ValueError('Output not found')
                output_string.append(int(all_output_names[c].split('.')[0][-1]))
                output_name.append(all_output_names[c])
                output_num.append(int(c))
                # print(f'added stim {s}/{unique_stims.size}')

            output_string = np.asarray(output_string)
            output_name = np.asarray(output_name)
            output_num = np.asarray(output_num)

            stim_num = []
            out_num = []
            stim_name = []

            for t in self.sweeps:
                stim_num.append(int(output_num[unique_stims==stim_tags[t]]))
                out_num.append(int(output_string[unique_stims==stim_tags[t]]))
                stim_name.append(output_name[unique_stims==stim_tags[t]])

            # save into a dict file
            out = {
                'stim_num': np.array(stim_num),
                'out_num': np.array(out_num),
                'stim_name': np.array(stim_name).squeeze().tolist(),
            }

        return out
    
    def info(self):
        """Display a the epoch structure and maybe in the future more information."""
        print(f'Mouse:           {self.mouse}')
        print(f'Date:            {self.date}')
        print(f'Virus:           {self.virus}')
        print(f'Genotype:        {self.genotype}')
        print(f'Imaging Params:  {self.imaging_params}')
        print(f'PMTs:            {self.pmts}')
        print('')
        # print(decode(self.path, 'ExpStruct/EpochText1')[self.epoch])
        # print(decode(self.path, 'ExpStruct/EpochText2')[self.epoch])
            
    @classmethod
    def load_file(cls, epoch, fr, rootdir='d:/frankenrig/experiments'):
        path = openfilegui(rootdir=rootdir, title='Select DAQ file')
        if not path:
            return
        print(f'Loaded DAQ file: {path}')
        return cls(path, epoch, fr)   
    
    def open_h5(self):
        h5 = h5py.File(self.path, 'r')
        self._h5 = h5
        return h5#['ExpStruct']
    
    def close_h5(self):
        if hasattr(self, '_h5'):
            self._h5.close()
            delattr(self, '_h5')
            
    def __del__(self):
        # ensure any open h5 is closed on delete
        self.close_h5()
        
    
    # def get_hsp_power

def count_daq_pulses(sweep):
    npulses = np.diff(sweep==1).sum()/2
    return np.int(npulses)

def parse_stim_data(stim_data):
    names = stim_data['stim_name']
    
    # split the name up by spaces
    # setupdaq uses the format 'Out 1: 50 mW 40Hz x40 10 ms wt 1 cell'
    conds = np.array([x for x in [n.split(' ')[2:5] for n in names]])
    pwr, hz, spks = tuple(map(np.array, conds.T))
    hz[hz=='No'] = '0'
    spks[spks=='Pulses'] = 'x0' # have to do this to split correctly later
    
    # pwr = np.array([''.join([i for i in n.split('mW')[0].split() if i.isdigit()]) for n in pwr], dtype=int)
    pwr = np.array([''.join([i for i in n.split('mW')[0].split()]) for n in pwr], dtype=float)
    hz = np.array([''.join([i for i in n.split('Hz')[0].split() if i.isdigit()]) for n in hz], dtype=int)
    spks = np.array([''.join([i for i in n.split('x')[1].split() if i.isdigit()]) for n in spks], dtype=int)
    
    
    out = {
        'power': pwr,
        'hz': hz,
        'spikes': spks,
        'cond': stim_data['out_num']
    }
    
    return out

def decode(daq_path, dataset):
    """
    Converts char array into readible strings from the daq.

    Inputs:
        dataset (str): '/' seperated directory path to to_dataset

    Returns:
        list of strings contained in the dataset
    """
    with h5py.File(daq_path, 'r') as f:
        try:
            strs = u''.join(chr(c) for c in f[dataset][:].squeeze())
            return strs
        except:
            # if there is an underlying dataset ref(s), go another level deep
            strs = [u''.join(chr(c[0]) for c in f[ref]) for ref in f[dataset][:].squeeze()]
            # this is a special case, but shouldn't hurt
            strs = [i.replace('\x00\x00',' ') for i in strs]
            return strs