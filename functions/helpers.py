# %%
import numpy as np
import pandas as pd
import itertools
import scipy.stats as st, chi2_contingency, chi2
from scipy.special import ndtri
from copy import deepcopy

# ground truth studyids
def get_id_dict_ACTUAL(severity_dict, sev_min=0, sev_max=4):
    """
    Retrieves a dictionary of IDs for each symptom within a specific severity range

    Parameters:
        severity_dict: dict 
            Dictionary containing symptom severity levels with corresponding patient IDs
        sev_min: int, optional
            Minimum severity level to include (default is 0).
        sev_max: int, optional
            Maximum severity level to include (default is 4).
    Returns:
        id_dict: dict
            Dictionary where keys are symptoms and values are lists of patient IDs with the specific severity range.
    """
    symptom_list = ['itching', 'cramp', 'dryskin', 'fatigue', 'musclesore'] # symptom list
    id_dict = {} # Initialize dictionary to store IDs
    for sym in symptom_list:
        # Get IDs for symptoms where severity is within the specific range
        ids = [severity_dict[sym][s] for s in severity_dict[sym] if (s >= sev_min) & (s <= sev_max)]
        ids = [*set(list(itertools.chain(*ids)))] # Removes duplicates
        id_dict[sym] = ids
    return id_dict


def get_ids_ACTUAL(severity_dict, sev_min=0, sev_max=4):
    """
    Retrieves a unique list of all patient IDs across symptoms within a specified severity range

    Parameters:
        severity_dict: dict 
            Dictionary containing symptom severity levels with corresponding patient IDs
        sev_min: int, optional
            Minimum severity level to include (default is 0).
        sev_max: int, optional
            Maximum severity level to include (default is 4).
    Returns:
        np.ndarray
            Array of unique patient Ids across all symptoms within the specified severity range.
    """
    symptom_list = ['itching', 'cramp', 'dryskin', 'fatigue', 'musclesore']
    ids_all = []
    for sym in symptom_list:
        # Get IDs for each symptom within severity range and add to list
        ids = [severity_dict[sym][s] for s in severity_dict[sym] if (s >= sev_min) & (s <= sev_max)]
        ids = [*set(list(itertools.chain(*ids)))]
        ids_all.extend(ids)
    return np.unique(ids_all)


# studyids reported by doc/nurse/nlp
def get_id_dict_byrater(symptom_dict, label):
    """
    Retrieves a dictionary of patients IDs reported by a specified rater for each symptom

    Parameters:
        symptom_dict: dict 
            Dictionary containing reported patient IDs by symptom and rater
        label: str
            Rater label, e.g., "doc", "nurse", "nlp", or "combined
    Returns:
        id_dict: dict
            Dictionary where keys are symptoms and values are lists of patient IDs with symptoms reported by the rater
    """
    symptom_list = ['itching', 'cramp', 'dryskin', 'fatigue', 'musclesore']
    id_dict = {}
    for sym in symptom_list:
        if label == "combined":
            ids_doc = symptom_dict[sym + '_' + "doc"]
            ids_nurse = symptom_dict[sym + '_' + "nurse"]
            id_dict[sym] = list(np.unique(ids_doc + ids_nurse))
        else:
            id_dict[sym] = symptom_dict[sym + '_' + label]
    return id_dict


# studyids reported by doc/nurse/nlp, regardless of symptom
def get_ids_byrater(symptom_dict, label):
    """
    Retrieves a unique list of all patient IDs reported by a specific rater across symptoms

    Parameters:
        symptom_dict: dict
            Dictionary containing reported patient IDs by symptom and rater
        label: str
            Rater label, e.g., "doc", "nurse", "nlp", or "combined
    Returns:
        np.ndarray
            Array of unique patient IDs reported by the specific rater across all symptoms
    """
    symptom_list = ['itching', 'cramp', 'dryskin', 'fatigue', 'musclesore']
    ids_all = []
    for sym in symptom_list:
        ids = symptom_dict[sym + '_' + label]
        ids_all.extend(ids)
    return np.unique(ids_all)


# confusion matrix for all symptoms combined
def get_confusion_matrix_ALL(pos_id_dict, id_dict, all_ids):
    """
    Calculates a confusion matrix (TP, FP, TN, FN) for all symptoms combined

    Parameters:
        pos_id_dict: dict
            Dictionary of actual positive patient IDs for each symptom.
        id_dict: dict
            Dictionary of predicted positive IDs for each symptom
        all_ids: np.ndarray
            Array of all unique Patient IDs
    Returns:
        dict
            Dictionary with counts of true positives (tp), false positives (fp), true negatives (tn), and false negatives (fn)
    """
    tp_total, fp_total, tn_total, fn_total = 0,0,0,0
    for sym in id_dict:
        pred_pos_ids = id_dict[sym]                            # predicted positive
        pred_neg_ids = np.setdiff1d(all_ids, pred_pos_ids)     # predicted negative
        pos_ids = pos_id_dict[sym]                             # patients positive for symptom (ground truth)
        neg_ids = np.setdiff1d(all_ids, pos_ids)               # patients negative for symptom (ground truth)
        
        tp_total += len(np.intersect1d(pred_pos_ids, pos_ids))
        fp_total += len(np.setdiff1d(pred_pos_ids, pos_ids))
        tn_total += len(np.intersect1d(pred_neg_ids, neg_ids))
        fn_total += len(np.setdiff1d(pred_neg_ids, neg_ids))
        
    return {'tp': tp_total, 'fp': fp_total, 'tn': tn_total, 'fn': fn_total}


# confusion matrix by symptom
def get_confusion_matrix(pos_ids, pred_pos_ids, all_ids):
    """
    Calculates a confusion matrix (TP, FP, TN, FN) for a single symptom

    Parameters:
        pos_ids: list
            List of actual positive patient IDs
        pred_pos_ids: list
            List of predicted positive patient IDs
        all_ids: np.ndarray
            Array of all unique patient IDs
    Returns:
        dict
            Dictionary with counts of true positives (tp), false positives (fp), true negatives (tn), and false negatives (fn)
    """
    tp_total, fp_total, tn_total, fn_total = 0,0,0,0

    pred_neg_ids = np.setdiff1d(all_ids, pred_pos_ids)  # predicted negative
    neg_ids = np.setdiff1d(all_ids, pos_ids)            # patients negative for symptom (ground truth)

    tp_total += len(np.intersect1d(pred_pos_ids, pos_ids))
    fp_total += len(np.setdiff1d(pred_pos_ids, pos_ids))
    tn_total += len(np.intersect1d(pred_neg_ids, neg_ids))
    fn_total += len(np.setdiff1d(pred_neg_ids, neg_ids))
        
    return {'tp': tp_total, 'fp': fp_total, 'tn': tn_total, 'fn': fn_total}


# true positive rate; probability of positive test result conditioned on individual being truly positive
# how well does test identify true positives?
def sensitivity(mat):
    """
    Calculates sensitivity (TP/(TP + FN))
    
    Parameters:
        mat: dict
            Confusion matrix with keys 'tp' and 'fn'
    Returns:
        float
            Sensitivity value or NaN if undefined
    """
    if (mat['tp'] + mat['fn']) == 0:
        return np.nan
    return mat['tp'] / (mat['tp'] + mat['fn'])


# true negative rate; probability of negative test result conditioned on individual being truly negative
# how well does test identify true negatives?
def specificity(mat):
    """
    Calculates specificty (TN / (TN + FP))

    Parameters:
        mat: dict
            Confusion matrix with keys 'tn' and 'fp'
    Returns:
        float
            Specificity value or NaN if undefined
    """
    if (mat['tn'] + mat['fp']) == 0:
        return np.nan
    return mat['tn'] / (mat['tn'] + mat['fp'])
    

# precision; number of true positive divided by number of positive calls
# if patient is marked positive for symptom, what is probability they actually report symptom?
def PPV(mat):
    """
    Calculates precision (TP / (TP + FP))

    Parameters:
        mat : dict
            Confusion matrix with keys 'tp' and 'fp'.
    Returns:
        float
            Positive Predictive Value or NaN if undefined.
    """
    if (mat['tp'] + mat['fp']) == 0:
        return np.nan
    return mat['tp'] / (mat['tp'] + mat['fp'])


# number of true negatives divided by number of negative valls
# if patient is marked negative for symptom, what is probability they actually don't have symptom?
def NPV(mat):
    """
    Calculates NPV (TN / (TN + FN)).

    Parameters:
        mat : dict
            Confusion matrix with keys 'tn' and 'fn'.

    Returns:
        float
            Negative Predictive Value or NaN if undefined.
    """
    if (mat['tn'] + mat['fn']) == 0:
        return np.nan
    return mat['tn'] / (mat['tn'] + mat['fn'])


def _proportion_confidence_interval(r, n, z):
    """Compute confidence interval for a proportion.

    Parameters:
        r: int
            Count of interest (e.g., true positives)
        n: int
            Total count (e.g., total positives)
        z: float
            Z-score for desired confidence level
    Returns:
        tuple
            Confidence interval for the proportion

    Follows notation described on pages 46--47 of [1]. 

    References
    ----------
    [1] R. G. Newcombe and D. G. Altman, Proportions and their differences, in Statisics
    with Confidence: Confidence intervals and statisctical guidelines, 2nd Ed., D. G. Altman, 
    D. Machin, T. N. Bryant and M. J. Gardner (Eds.), pp. 45-57, BMJ Books, 2000. 
    """

    if n==0:
        return (np.nan, np.nan)
    A = 2*r + z**2
    B = z*np.sqrt(z**2 + 4*r*(1 - r/n))
    C = 2*(n + z**2)
    return ((A-B)/C, (A+B)/C)


def confidence_interval(mat, alpha=0.95):
    """
    Calculates confidence intervals for sensitivity, specificity, PPV, and NPV.

    Parameters:
        mat: dict
            Confusion matrix with keys 'tp', 'fp', 'tn', 'fn'
        alpha: float, optional
            Confidence level (default is 0.95)
    Returns:
        tuple
            Confidence intervals for sensitivity, specificity, PPV, and NPV
    """
    TP = mat['tp']
    FP = mat['fp']
    TN = mat['tn']
    FN = mat['fn']
    
    z = -ndtri((1.0-alpha)/2)
    
    sensitivity_ci = _proportion_confidence_interval(TP, TP + FN, z)
    specificity_ci = _proportion_confidence_interval(TN, TN + FP, z)
    ppv_ci = _proportion_confidence_interval(TP, TP + FP, z)
    npv_ci = _proportion_confidence_interval(TN, TN + FN, z)
    
    return sensitivity_ci, specificity_ci, ppv_ci, npv_ci


def get_kappa(idlist1, idlist2, all_ids):
    """
    Calculates Cohen's Kappa for agreement between two raters.
    
    Parameters:
        idlist1: list
            List of IDs from rater 1
        idlist2: list
            List of IDs from rater 2
        all_ids: list
            List of all possible IDs
    Returns:
        float
            Kappa value indicating inter-rater agreement
    """
    # total number of instances that both raters said were correct
    A = len(np.intersect1d(idlist1, idlist2))
    # total number of instances rater 2 said was incorrect but rater 1 said was correct
    B = len(np.intersect1d(np.setdiff1d(all_ids, idlist2), idlist1))
    # total number of instances rater 1 said was incorrect but rater 2 said was correct
    C = len(np.intersect1d(np.setdiff1d(all_ids, idlist1), idlist2))
    # total number of instances both raters said were incorrect
    D = len(np.intersect1d(np.setdiff1d(all_ids, idlist1), np.setdiff1d(all_ids, idlist2)))

    # probability of agreement
    p_a = (A+D) / (A+B+C+D)
    p_correct = ((A+B)/(A+B+C+D)) * ((A+C)/(A+B+C+D))
    p_incorrect = ((C+D)/(A+B+C+D)) * ((B+D)/(A+B+C+D))
    p_e = p_correct + p_incorrect

    kappa = (p_a - p_e) / (1 - p_e)
    return kappa


def get_mcnemars(df_sym, label1, label2, alpha=0.01, display=False):
    """
    Performs McNemar's test to compare two raters' responses.
    
    Parameters:
        df_sym: pd.DataFrame
            DataFrame containing symptom ratings
        label1: str
            Column label for rater 1
        label2: str
            Column label for rater 2
        alpha: float, optional
            Significance level for the test (default is 0.01)
        display: bool, optional
            If True, displays the contingency table and test result (default is False)
    Returns:
        p_value: float
            p-value for McNemar's test
    """
    # create contingency table
    df_crosstab = pd.crosstab(df_sym[label1],
                                df_sym[label2],
                                dropna=False,
                                margins=True, margins_name="Total")
    df_crosstab = df_crosstab.reindex(index=[True,False], columns=[True,False], fill_value=0)
    
    # Calculation of McNemar's statistic
    rows = df_sym[label1].unique()
    columns = df_sym[label2].unique()
    mcnemar = (abs(df_crosstab[False][True] - df_crosstab[True][False]) - 1)**2 / \
                (df_crosstab[False][True] + df_crosstab[True][False])


    
    p_value = 1 - st.chi2.cdf(mcnemar, (len(rows)-1)*(len(columns)-1))
    conclusion = "Failed to reject the null hypothesis."
    if p_value <= alpha:
        conclusion = "Null Hypothesis is rejected."

    # The p-value approach
    if display:
        display(df_crosstab)
        print('\nComparing ' + label1 + ' and ' + label2)
        print("Approach 1: The p-value approach to hypothesis testing in the decision rule")
        print("McNemar's statistic is:", mcnemar, " and p value is:", p_value)
        print(conclusion)
    
    return p_value


# null hypothesis: observed frequencies for categorical variable match expected frequencies for categorical variable
# severity
# ignore individual symptoms

def pearson_chi2_v2(df_sym, rater, group, test='accuracy', alpha=0.01):
    """
    Conducts Pearson chi-square test on a symptom grouping based on accuracy

    Parameters:
        df_sym: pd.DataFrame
            DataFrame containing symptom data by raters
        rater: str
            Rater 
        group: str
            Grouping column 
        test: str, optional
            Test type to apply 
        alpha: float, otional
            Significance level (default is 0.01)
    Returns: 
        p: float
            p-value 
    """
    symptoms = ['fatigue', 'cramp', 'dryskin', 'musclesore', 'itching']
    # filter for finding ANY symptom
    filt_raters = [False]*len(df_sym)
    filt_actual = [False]*len(df_sym)
    for sym in symptoms:
        filt = df_sym[sym + '_' + rater]
        filt_raters = filt_raters | filt
        filt = df_sym[sym + '_actual']
        filt_actual = filt_actual | filt
        
    df_sym['accuracy'] = filt_actual == filt_raters
    # accuracy = filt_actual == filt_raters
    if test == 'sens':
        test_filt = filt_actual
    elif test == 'spec':
        test_filt = ~filt_actual
    elif test == 'ppv':
        test_filt = filt_raters
    elif test == 'npv':
        test_filt = ~filt_raters
    else:
        test_filt = [True]*len(df_sym)
    
    crosstab = pd.crosstab(df_sym.loc[test_filt, group], df_sym.loc[test_filt, 'accuracy'])
    display(crosstab)
        
    stat, p, dof, expected = chi2_contingency(crosstab)
    
    print('\nPearson test for rater ' + rater + ' grouped by ' + group)
    print('dof=%d' % dof)
    print(expected)
    
    # interpret test-statistic
    prob = 0.95
    critical = chi2.ppf(prob, dof)
    print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
    # interpret p-value
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
        
    return p
    


