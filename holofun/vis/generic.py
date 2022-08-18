import numpy as np
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
