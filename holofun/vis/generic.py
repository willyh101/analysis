import numpy as np
import pandas as pd
import scipy.stats as stats

def find_vis_resp(df, p=0.05, test='anova', vis_key='ori'):
    """
    Takes a mean dataframe (see meanby) and finds visually responsive cells using 
    a 1-way ANOVA test.
    
    Args:
        df (pd.DataFrame): mean dataframe (trials, cells, vis_condition)
        p (float, optional): p-valuse to use for significance. Defaults to 0.05.
        test (str, optional): statistical test to use, only one option now. Defaults to 'anova'.

    Returns:
        np.array of visually responsive cells
        np.array of p values for all cells
    """
    
    # for adding others later
    tests = {
        'anova': _vis_resp_anova(df, vis_key)
    }
    
    p_vals = tests[test]
    vis_cells = np.where(p_vals < p)[0]

    n = vis_cells.size
    c = p_vals.size
    print(f'There are {n} visually responsive cells, out of {c} ({n/c*100:.2f}%)')

    return vis_cells, p_vals

def po(mdf):
    """
    Takes a mean dataframe (see meanby) and returns preferred and
    orthagonal orientation in orientation space (mod 180).
    
    General procedure:
        1. Remove blank trial conditions (specified as -45 degs)
        2. Modulo 0-315* to 0-135* (mod excludes the number you put in)
        3. Get mean response by cell and orientation.
        4. Find index of max df, corresponding to PO.
        5. Subtract 90* from PO and mod 180 to get ortho

    Args:
        mdf (pd.DataFrame): mean response dataframe, generated from meanby (above)

    Returns:
        pd.Series of pref_oris
        pd.Series of ortho_oris
    """
    vals = mdf.loc[mdf.ori >= 0].copy()
    vals['ori'] = vals['ori'] % 180

    vals = vals.groupby(['cell', 'ori']).mean().reset_index()

    pref_oris = vals.set_index('ori').groupby('cell')['df'].idxmax()
    pref_oris.name = 'pref'
    
    ortho_oris = (pref_oris - 90) % 180
    ortho_oris.name = 'ortho'    

    return pref_oris, ortho_oris

def pdir(df):
    """Calculates pref dir."""
    df = df.loc[df.ori != -45]
    pref_dir = df.set_index('ori').groupby(['cell'])['df'].idxmax()
    pref_dir.name = 'pdir'

    return pref_dir

def osi(df):
    """
    Takes the mean df and calculates OSI.
    
    Procedure:
        1. Drop gray screen conditions (orientation == -45)
        2. Subtract off the minimum cell by cell. Note: it is VERY important do this
           to avoid negative values giving extremely high or low OSIs. Do this before
           averaging the tuning curves otherwise you get lots of OSIs = 1 (if ortho is
           is min and set to zero, OSI will always be 1).
        3. Groupby cell and ori to get mean dataframe/tuning curve.
        4. Get PO and OO values and calculate OSI.
    
    Returns a pd.Series of osi values
    
    Confirmed working by WH 7/30/20
    BUT LIKE REALLY REALLY FOR SURE THIS TIME
    
    """
    
    vals = df.loc[df.ori >= 0].copy()
    tc = vals.groupby(['cell', 'ori'], as_index=False).mean()
    
    osis = []
    for cell in vals.cell.unique():
        temp = tc[tc.cell == cell].copy()
        temp['df'] -= temp['df'].min()
        po = temp.df[temp.pref == temp.ori].values[0]
        o_deg1 = (temp.pref-90) % 360
        o_deg2 = (temp.pref+90) % 360
        o1 = temp.df[temp.ori == o_deg1].values[0]
        o2 = temp.df[temp.ori == o_deg2].values[0]
        oo = np.mean([o1, o2])
        osi = _osi(po,oo)
        osis.append(osi)

    osis = abs(np.array(osis))
    # osis = abs(osis[~np.isnan(osis)])

    osi = pd.Series(osi, name='osi')

    return osi

def _osi(preferred_responses, ortho_responses):
    """This is the hard-coded osi function."""
    return ((preferred_responses - ortho_responses)
            / (preferred_responses + ortho_responses))
    
def _global_osi(tuning_curve):
    # TODO
    pass

def _vis_resp_anova(data, vis_key):
    """Determine visual responsiveness by 1-way ANOVA."""

    f_val = np.empty(data.cell.nunique())
    p_val = np.empty(data.cell.nunique())

    for i,cell in enumerate(data.cell.unique()):
        temp3 = data[data.cell==cell]
        temp4 = temp3[[vis_key, 'trial', 'df']].set_index([vis_key,'trial'])
        samples = [col for col_name, col in temp4.groupby(vis_key)['df']]
        f_val[i], p_val[i] = stats.f_oneway(*samples)

    return p_val
