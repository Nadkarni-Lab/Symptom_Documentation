# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
sys.path.insert(0, '/Users/Symptom_Documentation/')
import helpers as hf

def plot_bygroup(symptom_dict, all_ids, actual_id_dict, color_dict, filename=None):
    """
    Plots sensitivity, specificty, PPV, and NPV for each rater group across cohorts. 

    Parameters:
        symptom_dict: dict
            Dictionary containing symptoms reported by patient, doctor, nurse and NLP
        all_ids: np.ndarray
            Array of all unique patient IDs
        actual_id_dict: dict
            Dictionary of ground truth IDs for each cohort
        color_dict: dict
            Dictionary of colors for each cohort
        filename: str, option
            Filename to save the plot if specified
    Return:
        plt.Figure
            Plot figure

    """
    rater_label_map = {'nurse': 'Nurse', 'doc': 'Physician', 'nlp': 'NLP'}

    # Severity low vs high

    plt.figure(figsize=[8, 1.5])
    plt.subplots_adjust(wspace=0.2, hspace=0)
    cols = 4
    rows = 1

    marker_size = 4
    raters = ['nlp', 'doc', 'nurse']
    cohort_symbol = ['o', 'o']

    for y, cohort in enumerate(actual_id_dict):
        for i, rater in enumerate(raters):
            ids_byrater = hf.get_ids_byrater(symptom_dict, rater)

            conf_mat = hf.get_confusion_matrix(actual_id_dict[cohort], ids_byrater, all_ids)

            sens = hf.sensitivity(conf_mat)
            spec = hf.specificity(conf_mat)
            ppv = hf.PPV(conf_mat)
            npv = hf.NPV(conf_mat)
            sens_ci, spec_ci, ppv_ci, npv_ci = hf.confidence_interval(conf_mat, alpha=0.95)

            y_offset = -y*0.1

            # Sensitivity
            plt.subplot(rows,cols,1)
            plt.plot(sens_ci, (i+y_offset,i+y_offset), color=color_dict[cohort])
            plt.plot(sens, i+y_offset, cohort_symbol[y], color='black', markersize=marker_size)
            plt.title('Sensitivity')
            plt.xlim([0, 1.05])
            plt.yticks([r for r in range(len(raters))], [rater_label_map[rater] for rater in raters])

            # Specificity
            plt.subplot(rows,cols,2)
            plt.plot(spec_ci, (i+y_offset,i+y_offset), color=color_dict[cohort])
            plt.plot(spec, i+y_offset, cohort_symbol[y], color='black', markersize=marker_size)
            plt.xlim([0, 1.05])
            plt.title('Specificity')

            #PPV
            plt.subplot(rows,cols,3)
            plt.plot(ppv_ci, (i+y_offset,i+y_offset), color=color_dict[cohort])
            plt.plot(ppv, i+y_offset, cohort_symbol[y], color='black', markersize=marker_size)
            plt.xlim([0, 1.05])
            plt.title('PPV')

            #NPV
            plt.subplot(rows,cols,4)
            if i==len(raters)-1:
                plt.plot(npv_ci, (i+y_offset,i+y_offset), color=color_dict[cohort], label=cohort)
            else:
                plt.plot(npv_ci, (i+y_offset,i+y_offset), color=color_dict[cohort])
            plt.plot(npv_ci, (i+y_offset,i+y_offset), color=color_dict[cohort])
            plt.plot(npv, i+y_offset, cohort_symbol[y], color='black', markersize=marker_size)
            plt.xlim([0, 1.05])
            plt.title('NPV')

            print(rater, '\t', cohort, '\tSENS:', np.round(sens, 2), '\tSPEC:', np.round(spec, 2), \
                  '\tPPV:', np.round(ppv, 2), '\tNPV:', np.round(npv, 2))

    plt.legend(loc="upper right", bbox_to_anchor=(2, 1.0))
    if filename:
        plt.savefig(filename, dpi=200,  bbox_inches='tight')
        plt.show()

    return plt.figure()
    
    
def plot_combined(symptom_dict, all_ids, actual_ids, filename=None):
    # physician survey,  nurse survey, NLP performance metrics
    # sensitivity, specificity, PPV, NPV
    """
    Plots combined performance metrics (sensitivity, specificty, PPV, NPV) for each rater group

    Parameters:
        symptom_dict: dict
            Dictionary containing symptoms reported by patient, doctor, nurse and NLP
        all_ids: np.ndarray
            Array of all unique patient IDs
        actual_ids: list
            List of ground truth positive IDs
        filename: str, optional
            Filename to save plot if specified
    Returns:
        plt.Figure
            Plot figure
        
    """
    
    rater_label_map = {'nurse': 'Nurse', 'doc': 'Physician', 'nlp': 'NLP', 'combined': 'Physician + Nurse'}

    plt.figure(figsize=[8, 1.5])
    plt.subplots_adjust(wspace=0.2, hspace=0)
    cols = 4
    rows = 1
    linecolor = '#c1272d'
    marker_size = 4

    raters = ['nlp', 'combined', 'doc', 'nurse']
    for i,rater in enumerate(raters):
        if rater == "combined":
            ids_doc = hf.get_ids_byrater(symptom_dict, "doc")
            ids_nurse = hf.get_ids_byrater(symptom_dict, "nurse")
            ids_byrater = np.union1d(ids_doc, ids_nurse)
        else:
            ids_byrater = hf.get_ids_byrater(symptom_dict, rater)  # patients reported by rater for ANY symptom

        conf_mat = hf.get_confusion_matrix(actual_ids, ids_byrater, all_ids)
        sens = hf.sensitivity(conf_mat)
        spec = hf.specificity(conf_mat)
        ppv = hf.PPV(conf_mat)
        npv = hf.NPV(conf_mat)
        sens_ci, spec_ci, ppv_ci, npv_ci = hf.confidence_interval(conf_mat, alpha=0.95)

        print(rater, '\tsens\t', np.round(sens, 3), '\t', np.round(sens_ci, 3))
        print(rater, '\tspec\t', np.round(spec, 3), '\t', np.round(spec_ci, 3))
        print(rater, '\tppv\t', np.round(ppv, 3), '\t', np.round(ppv_ci, 3))
        print(rater, '\tnpv\t', np.round(npv, 3), '\t', np.round(npv_ci, 3))

        plt.subplot(rows,cols,1)
        plt.plot(sens_ci, (i,i), color=linecolor)
        plt.plot(sens, i, 'o', color='black', markersize=marker_size)
        plt.title('Sensitivity')
        plt.xlim([0, 1.05])
        plt.yticks([r for r in range(len(raters))], [rater_label_map[rater] for rater in raters])

        plt.subplot(rows,cols,2)
        plt.plot(spec_ci, (i,i), color=linecolor)
        plt.plot(spec, i, 'o', color='black', markersize=marker_size)
        plt.xlim([0, 1.05])
        plt.title('Specificity')
        plt.yticks([])
        plt.tick_params(left = False)

        plt.subplot(rows,cols,3)
        plt.plot(ppv_ci, (i,i), color=linecolor)
        plt.plot(ppv, i, 'o', color='black', markersize=marker_size)
        plt.xlim([0, 1.05])
        plt.title('PPV')
        plt.yticks([])
        plt.tick_params(left = False)

        plt.subplot(rows,cols,4)
        plt.plot(npv_ci, (i,i), color=linecolor)
        plt.plot(npv, i, 'o', color='black', markersize=marker_size)
        plt.xlim([0, 1.05])
        plt.title('NPV')
        plt.yticks([])
        plt.tick_params(left = False)

    if filename:
        plt.savefig(filename, dpi=200,  bbox_inches='tight')
        plt.show()

    return plt.figure()

# %%
# physician survey,  nurse survey, NLP performance metrics
# sensitivity, specificity, PPV, NPV

def plot_bysymptom(symptom_dict, all_ids, actual_id_dict, filename=None):
    """
    Plots sensitivity, specificity, PPV, and NPV by symptom for each rater group.

    Parameters:
        symptom_dict: dict
            Dictionary containing reported symptom data by rater
        all_ids: np.ndarray
            Array of all unique patient IDs
        actual_id_dict: dict
            Dictionary of ground truth IDs for each symptom
        filename: str, optional
            Saves plot by specified filename if provided
    Returns:
        plt.Figure
            Resulting plot
    """
    
    symptom_label_map = {'cramp': 'Cramp', 'fatigue': 'Fatigue', 'musclesore': 'Muscle Soreness', \
                     'itching': 'Itching', 'dryskin': 'Dry Skin'}
    rater_label_map = {'nurse': 'Nurse', 'doc': 'Physician', 'nlp': 'NLP', 'combined': 'Physician + Nurse'}

    plt.figure(figsize=[8,4])
    plt.subplots_adjust(wspace=0.2, hspace=0)
    cols = 4
    rows = 1
    marker_size = 4

    raters = ['nurse','doc', 'combined', 'nlp']
    markers = ['o', 'o', 'o', 'o']
    colors = ['#0000a7', '#c1272d', '#eecc16', '#8c564b']
    symptom_order = ['fatigue', 'cramp', 'dryskin', 'musclesore', 'itching'][::-1]
    
    for i,rater in enumerate(raters):
        id_dict = hf.get_id_dict_byrater(symptom_dict, rater)
        
        # get stats by rater AND by symptom
        for j,sym in enumerate(symptom_order):
            conf_mat = hf.get_confusion_matrix(actual_id_dict[sym], id_dict[sym], all_ids)
            sens = hf.sensitivity(conf_mat)
            spec = hf.specificity(conf_mat)
            ppv = hf.PPV(conf_mat)
            npv = hf.NPV(conf_mat)
            sens_ci, spec_ci, ppv_ci, npv_ci = hf.confidence_interval(conf_mat, alpha=0.95)

            y = j-(i*0.1)
            marker = markers[i]

            ############ SENSITIVITY #######################
            plt.subplot(rows,cols,1)
            
            if j==len(id_dict)-1:
                plt.plot(sens_ci, (y,y), color=colors[i], label=rater_label_map[rater])
            else:
                plt.plot(sens_ci, (y,y), color=colors[i])
            plt.plot(sens, y, marker, color='black', markersize=marker_size)
            plt.title('Sensitivity')
            plt.xlim([-0.05, 1.05])
            plt.yticks([r for r in range(len(symptom_order))], \
                       [symptom_label_map[s] for s in symptom_order])

            ############ SPECIFICITY #######################
            plt.subplot(rows,cols,2)
            
            if j==len(id_dict)-1:
                plt.plot(spec_ci, (y,y), color=colors[i], label=rater_label_map[rater])
            else:
                plt.plot(spec_ci, (y,y), color=colors[i])
            plt.plot(spec, y, marker, color='black', markersize=marker_size)
            plt.xlim([-0.05, 1.05])
            plt.title('Specificity')
            plt.tick_params(left = False)
            plt.yticks([])

            ############ PPV #######################
            plt.subplot(rows,cols,3)
            
            if j==len(id_dict)-1:
                plt.plot(ppv_ci, (y,y), color=colors[i], label=rater_label_map[rater])
            else:
                plt.plot(ppv_ci, (y,y), color=colors[i])
            plt.plot(ppv, y, marker, color='black', markersize=marker_size)
            plt.xlim([-0.05, 1.05])
            plt.title('PPV')
            plt.tick_params(left = False)
            plt.yticks([])

            ############ NPV #######################
            plt.subplot(rows,cols,4)
            
            if j==len(id_dict)-1:
                plt.plot(npv_ci, (y,y), color=colors[i], label=rater_label_map[rater])
            else:
                plt.plot(npv_ci, (y,y), color=colors[i])
            plt.plot(npv, y, marker, color='black', markersize=marker_size)
            plt.xlim([-0.05, 1.05])
            plt.title('NPV')
            plt.tick_params(left = False)
            plt.yticks([])


    plt.legend(loc="upper right", bbox_to_anchor=(2.4, 1.0))
    
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.show()

    return plt.figure()

def plot_bysymptom_bygroup(symptom_dict, actual_id_dict, cohort_dict, color_dict, filename=None):
    """
    Plots sensitivity, specificity, PPV and NPV by symptom for each rater group within specific cohorts
    
    Parameters:
        symptom_dict: dict
            Dictionary containing reported symptoms data by rater
        actual_id_dict: dict
            Dictionary of ground truth IDs for each symptom
        cohort_dict: dict
            Dictionary containing IDs for each cohort
        color_dict: dict
            Dictionary of colors for each cohort
        filename: str, optional
            If provided, saves the plot with specified name
    Return:
        plt.Figure
            Resulting plot
    """
    # Define label mappings and marker settings
    symptom_label_map = {'cramp': 'Cramp', 'fatigue': 'Fatigue', 'musclesore': 'Muscle Soreness',
                         'itching': 'Itching', 'dryskin': 'Dry Skin'}
    rater_label_map = {'nurse': 'Nurse', 'doc': 'Physician', 'nlp': 'NLP'}
    symptom_order = ['fatigue', 'cramp', 'dryskin', 'musclesore', 'itching'][::-1]

    plt.figure(figsize=[8, 6])
    plt.subplots_adjust(wspace=0.2, hspace=0)
    cols = 4
    rows = 1
    raters = ['nlp', 'doc', 'nurse']  # Set specific rater order here
    markers = ['o', 's', 'v']
    marker_dict = dict(zip(raters, markers))  # Map each rater to a specific marker

    for c, cohort in enumerate(cohort_dict):
        for i, rater in enumerate(raters):
            id_dict = hf.get_id_dict_byrater(symptom_dict, rater)

            # Calculate stats by symptom for each cohort and rater
            for j, sym in enumerate(symptom_order):
                actual_ids = np.intersect1d(actual_id_dict[sym], cohort_dict[cohort])
                ids_byrater = np.intersect1d(id_dict[sym], cohort_dict[cohort])
                conf_mat = hf.get_confusion_matrix(actual_ids, ids_byrater, cohort_dict[cohort])
                
                sens = hf.sensitivity(conf_mat)
                spec = hf.specificity(conf_mat)
                ppv = hf.PPV(conf_mat)
                npv = hf.NPV(conf_mat)
                sens_ci, spec_ci, ppv_ci, npv_ci = hf.confidence_interval(conf_mat, alpha=0.95)

                y_offset = -c * 0.08
                y = j + (i * 0.2) - 0.2 + y_offset
                marker = marker_dict[rater]

                # Plot sensitivity
                plt.subplot(rows, cols, 1)
                plt.plot(sens_ci, (y, y), color=color_dict[cohort])
                plt.plot(sens, y, marker, color='black')
                plt.title('Sensitivity')
                plt.xlim([-0.05, 1.05])
                plt.yticks([r for r in range(len(symptom_order))],
                           [symptom_label_map[s] for s in symptom_order])

                # Plot specificity
                plt.subplot(rows, cols, 2)
                plt.plot(spec_ci, (y, y), color=color_dict[cohort])
                plt.plot(spec, y, marker, color='black')
                plt.xlim([-0.05, 1.05])
                plt.title('Specificity')
                plt.tick_params(left=False)
                plt.yticks([])

                # Plot PPV
                plt.subplot(rows, cols, 3)
                plt.plot(ppv_ci, (y, y), color=color_dict[cohort])
                plt.plot(ppv, y, marker, color='black')
                plt.xlim([-0.05, 1.05])
                plt.title('PPV')
                plt.tick_params(left=False)
                plt.yticks([])

                # Plot NPV
                plt.subplot(rows, cols, 4)
                if j == 0:
                    plt.plot(npv_ci, (y, y), color=color_dict[cohort], label=cohort)
                    plt.plot(npv, y, marker, color='black', label=rater_label_map[rater])
                else:
                    plt.plot(npv_ci, (y, y), color=color_dict[cohort])
                    plt.plot(npv, y, marker, color='black')
                plt.xlim([-0.05, 1.05])
                plt.title('NPV')
                plt.tick_params(left=False)
                plt.yticks([])

    # Create custom legend handles, ordered by cohort and then by specified rater order
    custom_legend_handles = [
        Line2D([0], [0], color=color_dict[cohort], marker=marker_dict[rater], markersize=6, 
               markerfacecolor='black', markeredgecolor='black', linestyle='-', 
               label=f"{rater_label_map[rater]} {cohort}")
        for cohort in reversed(cohort_dict)
        for rater in raters 
    ]

    custom_legend_handles = custom_legend_handles[::-1]

    # Set the legend with custom handles
    plt.legend(handles=custom_legend_handles, loc="upper right", bbox_to_anchor=(2.65, 1.0))

    if filename:
        plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.show()

    return plt.figure()


# %%



