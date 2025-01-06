#Import required libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import csv

"""This program gets the actual deposition of the devices
Inputs: Raw data file, Spreadsheet info passed by deposition_update.py
Outputs: Graph of compositions (passed by name), which can be uploaded by deposition_update.py
"""

def get_composition_graphs(file_name, local_file_save_path, content, mat_names, thicknesses, zero_counts):
    #Get required Data: Need total thickness of each  material, data logs (all materials), number of materials

    num_materials = len(mat_names)
    total_thickness = sum(thicknesses)
    start_time = zero_counts[len(zero_counts)-2] #At what time does thickness start recording?
    end_time = zero_counts[len(zero_counts)-1] #At what time does thickness stop recording?
    total_time = end_time - start_time
    #print("Total depositing time:", total_time)

    #Column lookups (will be based on number of materials)
    rate_base_col = 1
    rate_data = []
    for i in range (0, num_materials):
        rate_data.append(np.transpose(content)[i+rate_base_col])

    #Adjust data using thickness correction... Perhaps I can make use of existing python file... Do later

    #Establish variables:
    #print ("Layer total thickness: ", total_thickness)
    num_bins = 10 #Number of x-axis stages on histogram
    stage_thickness = 0.01 #Thickness of a given stage
    num_stages = int(total_thickness / stage_thickness) #Must round into an integer
    #print("Number of Stages:", num_stages)
    stage_time = int(total_time / num_stages)
    #print ("Stage time in seconds: ", stage_time)

    #Iterate through the data... When thickness at each step exceeds bin_thickness, move to next step
    #Only iterate through data after 2nd to last zero of the host
    #Determine the %host %fd and %td in each step % = component_stage_thickness / total_stage_thickness
    #Ultimately we want a set of %host %fd and %td for each bin, and we want to plot these as a histogram
    comp_data = []
    for i in range (0, num_materials):
        comp_data.append([])
    for stage_num in range (0, num_stages):
        stage_start = start_time + stage_num*stage_time #When does this stage start?
        stage_end = stage_start + stage_time #When does this stage end?

        if (stage_end < end_time): #Only go if we don't overrun on our data
            avg_rates = []
            for i in range (0, num_materials):
                avg_rates.append(np.average(rate_data[i][stage_start:stage_end].astype(float)))

            comp_data_stage = []
            for i in range (0, num_materials):
                comp_mat = avg_rates[i] / np.sum(np.array(avg_rates))
                comp_data[i].append(comp_mat)

    fig, axs = plt.subplots(1, num_materials, sharey=True, tight_layout=True, figsize=(5*num_materials,5))

    dispersity = []
    # We can set the number of bins with the `bins` kwarg
    for i in range (0, num_materials):
        x = comp_data[i][1:]
        x = [a for a in x if a > 0.001] #Get rid of bad values
        if (len(x) == 0):
            x = [0, 0.0001]
        avg_comp = np.average(x)
        wt_avg_comp = np.average(x, weights=x)
        dispersity.append(wt_avg_comp / avg_comp)
        axs[i].hist(x, bins=num_bins)
        axs[i].set_xlabel("%Composition")
        axs[i].set_ylabel("Number of 10 Angstrom Intervals")
        axs[i].set_title(mat_names[i] + ": Avg="+ str(avg_comp*100)[0:5]+"%, PDI="+ str(dispersity[i])[0:6])

    uniformity_score = np.sum(dispersity) - num_materials
    fig.suptitle(file_name + ": Composition Distribution - Uniformity Score: " + str(uniformity_score)[0:6] + " (0 is ideal)")
    plot_name = file_name + " Composition.png"
    plt.savefig(local_file_save_path + "/" + plot_name) #Saves the data
    plt.clf()
    return plot_name
