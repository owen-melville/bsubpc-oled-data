#NOTE: This requires the cols_needed file to identify what columns you need {requires exact match for column lookup}
#You can edit cols_needed, but make sure to keep thickness, power, and rate or the program will fails
import csv
import matplotlib.pyplot as plt
import numpy as np

#process raw data and return condensed data
def get_condensed_data(file_path, mat_names, sensor_nums):
    content = [] #master data
    num_materials = len(mat_names)

    #Fetch the first row to do column lookups
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            if len(row) > 1:
                first_row = row
                break

    #This gets the columns you want to get from an input file
    with open ('cols_needed.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            try:
                col_index = first_row.index(row[0])+1 #Lookup the column index from the first row
            except:
                col_index = None
            col_name = row[0]

            if (col_name == 'Timestamp'):
                included_cols = [0] #Timestamp is always the 0th column and isn't labelled
                included_col_names = ['Timestamp'] #Add the timestamp column

            elif (col_name == 'Fundamental  Frequency' or col_name == 'Current Z-Ratio'): #Have to get columns using sensor numbers here
                col_index -= 1 #Just an adjustment for the sensor based ones
                for mat in range(0, num_materials): #Essentially, we need to get columns for each material
                    included_cols.append(col_index+int(sensor_nums[mat]))
                    included_col_names.append(col_name+": " + mat_names[mat])
            else:
                for mat in range(0, num_materials): #Most other columns are done by material not by sensor number
                    included_cols.append(col_index+mat)
                    included_col_names.append(col_name+": " + mat_names[mat])
    content.append(included_col_names) #Add column headers

    #This basically determines the maximum row length to allow padding of short rows
    max_len = 0
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            if len(row) > max_len:
                max_len = len(row)

    #This creates a list of lists with the data you want called content, while padding short rows to allow proper extraction
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        rows = 0
        for row in reader:
            if (len(row) > 1): #skip non-columnar data
                if (len(row) < max_len):
                    for i in range (0,max_len-len(row)): #Make all the rows the same length
                        row.insert(86, 0) #86 is a good place to but in buffer zeroes
                content.append(list(row[i] for i in included_cols)) #only gets the columns you want #Error here when the columns change (eg freq)
            rows+=1
    return content

def get_material_thicknesses_and_plotnames(mat_names, content, deposition_id, file_save_path):
    return_data = [] #Data to be returned by method
    num_materials = len(mat_names)
    zero_counts = []
    #For each material, calculate thickness and create graphs
    #This part will fail right now if the columns are not as expected. Later I will fix this.
    #Thickness calculation is essentially thickness = final_thickness - initial_thickness + thicknesses_before_reset
    #In order to detect the resets, we need to set a thickness drop threshold, a small value that can't occur from fluctuations
    for mat in range(0,num_materials):
        mat_name = mat_names[mat]
        t_prev = 0
        p_prev = 0
        t_total = 0
        t_starting = 0
        t_deposited = 0
        drop_threshold = 0.003 #The program considers a drop of 3A a sign that thickness was reset
        min_power_thickness_threshold = 0.005 #The program considers 5A the onset of rate for determining the onset power
        onset_power = 0 #power at which we see rate (for scaling y-axis)
        min_power = 0 #sometimes, the power dips below onset power
        power_dip_threshold = 1.5 #How far below the onset_power will the y-axis fall if the power dips
        power_jump_threshold = 0.2 #What dip would be considered cutting off the power?
        pow_recorded = False
        max_power = 0
        count = 0
        rate = []
        time = []
        power = []
        thickness = []

        #Parse through the data by row
        #This process-based analysis is good for determining "points" such as the end of the deposition as indicated by a zero
        #Some of these calculations are a bit finicky because they are meant to detect human actions from data
        for row in content:
            if count==0:
                #Get the columns for this particular material
                #This part will fail if cols_needed doesn't include these columns (should do a return clause)
                try:
                    thickness_index = row.index("Thickness: " + mat_names[mat])
                    power_index = row.index("Source Power: " + mat_names[mat])
                    rate_index = row.index("Filtered Rate: " + mat_names[mat])
                except:
                    print("Error getting thickness/power/rate columns")
                    return None
            if count!=0: #exclude headers
                #Extract the thickness, power, and rate
                t_current = float(row[thickness_index])
                p_current = float(row[power_index])
                r_current = float(row[rate_index])

                if (count==1):
                    t_total -= t_current #subtract starting thickness
                    t_starting = t_current
                if (np.abs(t_current - t_prev) > drop_threshold and np.abs(t_current) < 0.005 and mat==0): #Check for drop for actual thickness
                    zero_counts.append(count)
                    t_deposited = t_prev
                if (t_current == 0 or count==len(content)-1 or t_current < t_prev - 0.005) and t_prev > 0: #make sure to add final thickness
                    t_total += t_prev
                    t_prev = t_current
                if (t_current - t_starting > min_power_thickness_threshold and pow_recorded == False):
                    min_power = p_current
                    onset_power = p_current
                    pow_recorded = True
                if (p_current > max_power and p_current < p_prev + power_jump_threshold):
                    max_power = p_current
                #This logic is trying to allow the min_power to dip without dipping too much
                #If we are 1) lower than previously recorded and 2) aboove a set amount below onset and 3) not jumping rapidly downwards
                if (p_current < min_power and p_current > onset_power - power_dip_threshold and p_current > p_prev - power_jump_threshold):
                    min_power = p_current
                time.append(count) #Assuming that each step is 1s (adjust this later to actually check this from timestamp
                rate.append(r_current) #Rate vector
                power.append(p_current) #Power vector
                thickness.append(t_current) #Thickness vector
                p_prev = p_current
                r_prev = r_current
                t_prev = t_current
            count+=1
        if len(zero_counts) == 0:
            zero_counts.append(1)
        if len(zero_counts) == 1:
            zero_counts.append(len(content))
        print("Zeroes: ", zero_counts)
        avg_rate = 0
        rate_stdev = 0
        avg_power = 0
        try:
            end_count = zero_counts[-1] #When we stop counting
            start_count = zero_counts[len(zero_counts)-2] #When we start counting
            if (t_deposited == 0):
                t_deposited = t_total
            if mat > 0 and end_count > 0: #If codepositing
                t_deposited = float(content[end_count][1+2*num_materials+mat])
                back = 0
                while t_deposited == 0:
                    t_deposited = float(content[end_count-back][1+2*num_materials+mat])
                    back+=1

            rates = [float(k) for k in [j[1+mat] for j in content[start_count:end_count]]]
            avg_rate = sum(rates) / len(rates)
            rate_stdev = np.std(rates)

            powers = [float(k) for k in [j[1+num_materials+mat] for j in content[start_count:end_count]]]
            avg_power = sum(powers) / len(powers)

            time_deposited_sub = end_count - start_count
            total_time = len(content)/60
            if (mat==0):
                print("Total time logging deposition: " +  str(total_time)[0:4] + " minutes")
                print("Total time spent depositing on substrate: " + str(time_deposited_sub) + " seconds")
 
            low_thickness_threshold = 0.025 #this is the threshold below which we do the finicky calculation of thickness to avoid rounding issues
            if t_deposited < low_thickness_threshold and end_count > 0: #If the thickness is really low
                last_change_count = 0
                for i in range (0,end_count): #We count down from "end point"
                    if last_change_count == 0:
                        t_current = float(content[end_count-i][1+2*num_materials+mat])
                        if t_current == t_deposited-0.001: #When we hit 1 Angstrom lower
                            last_change_count = end_count-i
                            rates = [float(k) for k in [j[1+mat] for j in content[last_change_count:end_count]]]
                            avg_rate = sum(rates) / len(rates) #Get average rates
                            rate_stdev = int(np.std(rates)*1000)/1000 #Avoid rounding errors
                            t_deposited = t_deposited - 0.0005 + avg_rate*(i-1)/1000 #Minus 1 because we've gone too far
                    i += 1
        except Exception as e:
            t_deposited = 0
            print(e)


        #PLOT!!! Single plot for each material looks nice and clean
        #Note: The power axis is not auto-scaled because I want to zoom in on where we actually have Rate
        #Sometimes, this scaling results in weird looking graphs, particularly if the power was dropped below onset power to drop the rate quickly
        plt.style.use('seaborn-whitegrid')
        breathing_room = np.maximum(0.01, (max_power-min_power)*0.10) #Give 10% space to graph
        ax1 = plt.subplot(3,1,1)
        color = 'tab:red'
        ax1.set_ylabel('Power (%)', color=color)
        ax1.plot(time, power, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(min_power-breathing_room, max_power+breathing_room)
        plt.title("Run: " + deposition_id + ', Material: ' + mat_names[mat] + ", Total thickness = " + str(t_total)[:5] + " kA")
        ax2 = plt.subplot(3,1,2)
        color = 'tab:blue'
        ax2.set_ylabel('Rate (A/s)', color=color)
        ax2.plot(time, rate, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, np.amin([np.amax(rate), avg_rate*2])*1.1)
        ax3 = plt.subplot(3,1,3)
        color = 'tab:green'
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Thickness (kA)', color=color)
        ax3.plot(time, thickness, color=color)
        ax3.tick_params(axis='y', labelcolor=color)
        plot_name = deposition_id +'_'+mat_names[mat]+ '.png'
        plt.savefig(file_save_path+"/"+plot_name)
        plt.clf()
        return_data.append([t_total, t_deposited, plot_name, avg_rate, rate_stdev, avg_power, total_time, time_deposited_sub])
    return return_data, zero_counts #data returned to deposition_update.py
