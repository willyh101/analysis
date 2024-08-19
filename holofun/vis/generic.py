import numpy as np
import pandas as pd
import scipy.stats as stats

from ..analysis import _ttest_rel_df, _wilcoxon_df, _increase_test_df, _ranksum_df, df_add_cellwise

def find_vis_resp(df, p=0.05, test='anova', vis_key='ori', quiet=False, **kwargs):
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
    if not quiet:
        print(f'There are {n} visually responsive cells, out of {c} ({n/c*100:.2f}%) <across conds>')

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

def find_vis_resp_pre_post(df: pd.DataFrame, win: tuple[float], 
                           one_way=False, test_fxn='ttest', alpha=0.05):
    test_funs = {
        'ttest': _ttest_rel_df,
        'wilcoxon': _wilcoxon_df,
        'increase': _increase_test_df,
        'ranksum': _ranksum_df,
    }
    test = test_funs[test_fxn]

    base = df[(df.time > win[0]) & (df.time < win[1])].groupby(['cell', 'trial']).mean(numeric_only=True).reset_index()
    resp = df[(df.time > win[2]) & (df.time < win[3])].groupby(['cell', 'trial']).mean(numeric_only=True).reset_index()['df']

    base = base.rename(columns={'df':'baseline'})
    resp = resp.rename('resp')
    
    resp_df = pd.concat((base,resp), axis=1)

    ps = resp_df.groupby('cell').apply(lambda x: test(x)).values
    
    if one_way:
        ps /= 2
    
    resp_df = df_add_cellwise(resp_df, ps, 'pval')
    cells = resp_df[resp_df['pval'] < alpha].cell.unique()
    # cells = np.where(ps < pval)[0]

    n = cells.size
    c = ps.size
    print(f'There are {n} visually responsive cells, out of {c} ({n/c*100:.2f}%) <pre/post>')
    
    return cells, ps