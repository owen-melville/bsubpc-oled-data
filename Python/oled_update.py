#A program to update everything
from os import path
import numpy as np
from datetime import datetime
import csv
import os
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import pandas as pd
import pathlib
import math
from scipy.signal import savgol_filter
from openpyxl import load_workbook
from string import ascii_uppercase

#Where the output data lives
OUTPUT_DATA_SHEET = 'Deposition Output'
OUTPUT_DATA_COLUMNS = 'A:M'

#Where material properties lives
ENERGY_DATA_SHEET = 'Energetic Properties'
ENERGY_DATA_COLUMNS = 'A:G'

 #Where data about the IC6 / physical tooling data lives
MATERIAL_DATA_SHEET = 'Evaporator Materials'
MATERIAL_DATA_COLUMNS = 'A:S'

 #Access the Master Deposition Spreadsheet
MASTER_DEPOSITION_FILE = 'C:/Users/AM/Desktop/Offline Spreadsheets/Master Deposition List.xlsx'

#OLED Spreadsheet Data
OLED_INPUT_SHEET = 'OLED Input'
OLED_INPUT_COLUMNS = 'A:E'

OLED_OUTPUT_SHEET = 'OLED Output'
OLED_OUTPUT_COLUMNS = 'A:Q'

MASTER_OLED_FILE = 'C:/Users/AM/Desktop/Offline Spreadsheets/Master OLED Log.xlsx'

OLED_PIXEL_NAMES = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4", "D5"]
WAVELENGTHS = np.arange(380,780,0.5)

createFiles = True
create_opto_graphs = True

def main():
    print("OLED Update Running...")
    print("Created by Owen Melville, 2023")
    print("See Python Instructions File for Details")

    #Get Deposition Data
    deposition_output_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, OUTPUT_DATA_SHEET, OUTPUT_DATA_COLUMNS)

    #Get Material Data for example HOMO and LUMO energies
    material_data_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, ENERGY_DATA_SHEET, ENERGY_DATA_COLUMNS)

    #Get Evaporator Data (including tooling)
    evaporator_data_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, MATERIAL_DATA_SHEET, MATERIAL_DATA_COLUMNS)

    #Get OLED Input and Output Data
    oled_input_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_INPUT_SHEET, OLED_INPUT_COLUMNS)
    oled_output_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_OUTPUT_SHEET, OLED_OUTPUT_COLUMNS)

    #Checks how many depositions have been done
    print("OLED Devices Fabricated According to OLED Master Log:", len(oled_input_df))

    #How many depositions have been characterized?
    cells_with_data = oled_input_df.loc[oled_input_df['Obtained Optical Data? (Y/N)']=='Y']
    print("OLED Devices Characterized According to OLED Master Log:", len(cells_with_data))

    #How many devices have been processed?
    print("OLED Devices Processed: ", len(oled_output_df))

    #Lets check the processing status of every device
    rows_to_edit = []
    for input_index, input_row in oled_input_df.iterrows():
        if input_row['Obtained Optical Data? (Y/N)']=='Y':
            device_status = 'New' #Default
            device_id = input_row['OLED ID']
            for output_index, output_row in oled_output_df.iterrows():
                if device_id==output_row['OLED ID']:
                    device_status = output_row['Status'] #We find an existing device with some status
                    break
           
            if device_status == 'New': #Create new row
                print("Found new device: ", device_id)
                new_row = {'OLED ID':device_id,  'Status':'Unprocessed'}
                oled_output_df.loc[len(oled_output_df)] = new_row #Add the new row
                oled_output_df = oled_output_df.where(pd.notnull(oled_output_df), "TBD") #Replace NaN values with TBD
                device_status = 'Unprocessed'
            
            if device_status != 'Processed': #Process any unprocessed row
                print("\nProcessing device: ", device_id)

                output_index = oled_output_df.loc[oled_output_df['OLED ID'] == device_id].index[0]
                rows_to_edit.append(output_index)

                #1. check if raw data exists
                raw_data_folder_path = str(pathlib.Path().resolve()).replace("Python", "Raw OLED Data") + "/" + device_id
                if path.exists(raw_data_folder_path):
                    print("Folder found: ", raw_data_folder_path)
                   
                    #Check if raw data exists in the form of files
                    oled_files = os.listdir(raw_data_folder_path)
                    filtered_oled_files = []

                    #Find the files whose names match A1, A2 and that are the relevant file (not the counts or SPD file)
                    for file_name in oled_files:
                        if file_name.split(".")[0].split("-")[0] in OLED_PIXEL_NAMES and 'counts' not in file_name and 'SPD' not in file_name:
                            filtered_oled_files.append(file_name)
                    print("Recognized Files (Tested Pixels): ", len(filtered_oled_files))
                    
                     #convert raw data into compressed data
                    if len(filtered_oled_files)>0:
                        compressed_data = create_compressed_oled_data(raw_data_folder_path, filtered_oled_files)
    
                        #Create a directory if it doesn't exist
                        local_file_save_path = str(pathlib.Path().resolve()).replace("Python", "Output Data") + "/" + device_id
                        if path.exists(local_file_save_path):
                            print("Local output directory already exists...")
                        else:
                            print("Creating Local Output Directory: " +device_id)
                            os.mkdir(local_file_save_path)

                        #Create a directory for the data
                        local_data_save_path = local_file_save_path + '/Raw Data'
                        if path.exists(local_data_save_path):
                            print("Local data directory already exists...")
                        else:
                            print("Creating Local Data Directory: " +device_id)
                            os.mkdir(local_data_save_path)

                        #0a. Save compressed data locally
                        compressed_data.to_csv(local_data_save_path+"/"+device_id+"_condensed_oled_data.csv", sep='\t', encoding='utf-8')

                        #0b. Filter away duplicate pixels
                        compressed_data_filtered = filter_compressed_data(compressed_data)

                        #0c. Get Working Pixels from compressed data
                        working_devices = compressed_data_filtered[compressed_data_filtered['Status'].str.contains('Working')]

                        #use compressed data to generate summary sheets
                        #1. OLED Stack Png (Requires materials, energy levels, and thicknesses) 
                        try: 
                            generate_oled_stack_png(device_id, material_data_df, deposition_output_df, local_file_save_path)
                            print("Sucessfully Generated OLED Stack Image")
                        except Exception as err:
                            print("Could not generate OLED stack image")
                            print(err)

                        #2. CIE Scatter Plot (Requires OLED results only)
                        create_CIE_scatter_plot(working_devices, device_id, local_file_save_path)

                        #3. Visual Reports (Requires OLED results)
                        create_visual_report(compressed_data_filtered, device_id, local_file_save_path)

                        #4. Stack Reports (Requires tooling planes, deposition data)
                        stack_df = create_stack_report(deposition_output_df, evaporator_data_df, device_id, local_file_save_path)

                        #5. Histograms
                        create_performance_histograms(working_devices, device_id, local_file_save_path)

                        #6. Correlation Analysis (Graphs of features vs. performance)
                        correlation_row = check_correlations(working_devices, stack_df,device_id, local_file_save_path)

                        #7. Get average values for spreadsheets
                        output_row, median_color = update_oled_output_data(compressed_data_filtered, device_id, correlation_row)
                        oled_output_df.loc[output_index] = output_row

                        if create_opto_graphs:
                            #8. Summary Optoelectronic Data + Graphs
                            working_pixels = create_summary_optoelectronic_graphs(working_devices, device_id, local_file_save_path, raw_data_folder_path, median_color)

                            #9. Save Working Pixels:
                            working_pixels = working_devices['Pixel']
                            working_pixels.to_csv(local_data_save_path+"/"+device_id+"_working_pixels.csv", sep='\t', encoding='utf-8')
                    else:
                        print("No Files Found")
                        oled_output_df.loc[output_index, "Status"] = "No Files Found"
                else:
                    print("Folder not found")
                    oled_output_df.loc[output_index, "Status"] = "Folder Not Found"

    #upload = [oled_output_df.columns.values.tolist()] + oled_output_df.values.tolist()
    #update_spreadsheet(sheets_service, OLED_FILE_ID, OLED_OUTPUT_RANGE, {'values': upload})
    convert_df_to_xlsx(MASTER_OLED_FILE, OLED_OUTPUT_SHEET, OLED_OUTPUT_COLUMNS, oled_output_df, rows_to_edit, True)

def create_performance_histograms(summary_data, device_id, local_file_save_path):
    print("Making Histograms...")
    HISTOGRAM_CATEGORIES = ['CE(cd/A)_max', 'CE(cd/A)_avg']
    for category in HISTOGRAM_CATEGORIES:
        fig, ax = plt.subplots()
        x = summary_data[category]
        plt.hist(x)
        plt.xlabel(category)
        plt.ylabel('Counts')
        label = device_id+'_hist_'+ category.split('(')[0]+category.split(')')[-1]+'.png'
        plt.savefig(local_file_save_path+"/"+label)

def create_summary_optoelectronic_graphs(working_devices, device_id, local_file_save_path, raw_data_path, median_color):
    print("Creating optoelectronic summary graphs...")

    #These are the categories we want to make graphs from
    RESULT_CATEGORIES = ['Voltage(V)',  'Current(A)',  'J(mA/cm2)',   'PE(lm/W)', 'EQE(%)',  'L(cd/m2)', 'CE(cd/A)'] #Note, added category
    #How we want to plot these categories
    RESULT_SCALE = ['linear', 'log', 'log', 'linear', 'linear', 'log', 'linear']
    #Minimum values that we want to plot for these values
    MIN_VALUE = [0, 1E-5, 1, 0, 0,1,0]
    DESIRED_PAIRINGS = [[0,2,False],[2,5,False], [2,4,True], [2,3,True],[2,6,True],[0,5,False],[5,3,True], [5,4,True],[5,6,True]]

    #Create a directory if it doesn't exist
    local_graph_save_path = local_file_save_path + '/Parameter Graphs'
    if path.exists(local_graph_save_path):
        print("Local output directory already exists (Graphs)...")
    else:
        print("Creating Local Output Directory (Graphs): " +device_id)
        os.mkdir(local_graph_save_path)

    #This code section is to allow calculation even if we don't meet the threshold
    DEFAULT_COMPARISON_LUMINANCE = [250, 100] 
    luminance_threshold = 0
    for default_lum in DEFAULT_COMPARISON_LUMINANCE:
        luminance_above_threshold_list = working_devices['Max Luminance (cd/m2)'].values > default_lum
        num_above = sum(bool(x) for x in luminance_above_threshold_list) 
        if num_above > len(working_devices)*0.5:
            luminance_threshold = default_lum
            break

    working_devices_above_threshold = working_devices[working_devices['Max Luminance (cd/m2)'] > luminance_threshold]
    working_pixels_above_threshold = working_devices_above_threshold['Pixel'].values

    working_pixels = working_devices['Pixel'].values
    on_voltage = np.median(working_devices['Turnon Voltage (V)'].values)

    lum_voltages = working_devices['V (V) @ '+str(luminance_threshold)+' cd/m2'].values

    all_pixel_pd = pd.DataFrame(columns=['Pixel']+RESULT_CATEGORIES)
    
    all_pixel_summary_data = []
    all_pixel_spectral_data = np.empty((len(working_pixels_above_threshold), len(WAVELENGTHS)))
    lengths = []
    for i in range (0, len(working_pixels)): #Get the raw data
        pixel = working_pixels[i]
        
        summary_data_file = pixel + ".txt"

        pixel_input = pd.read_csv(raw_data_path+'/'+summary_data_file, delimiter='\t')
        pixel_input['CE(cd/A)']=pixel_input['L(cd/m2)']/pixel_input['J(mA/cm2)']/10 #Calculating the luminescent efficiency
        pixel_summary = pixel_input.to_numpy()
        pixel_input['Pixel']=pixel
        
        all_pixel_pd = pd.concat([all_pixel_pd, pixel_input], ignore_index=True)
        lengths.append(pixel_summary.shape[0])
        all_pixel_summary_data.append(pixel_summary)

    for i in range (0, len(working_pixels_above_threshold)):
        pixel = working_pixels_above_threshold[i]
        device_data = working_devices[working_devices['Pixel']==pixel]
        spectral_data_file = pixel + "_counts.txt"
        
        pixel_index = list(working_pixels).index(pixel)
        pixel_summary = all_pixel_summary_data[pixel_index]
        lum_voltage = lum_voltages[pixel_index]

        try:    
            lum_index = next(x[0] for x in enumerate(pixel_summary[:, 0]) if x[1] > lum_voltage) + 2 #Plus 2 is because of the 2 columns of 0s in the file
        except:
            lum_index = len(pixel_summary[:,0]+2)

        pixel_spectra = pd.read_csv(raw_data_path+'/'+spectral_data_file, delimiter='\t').to_numpy()[:, lum_index]

        all_pixel_spectral_data[i, :] = pixel_spectra

    all_pixel_pd.to_csv(local_file_save_path+"/Raw Data/"+device_id+"_all_optoelectronic.csv", sep='\t', encoding='utf-8')

    min_length = np.amin(lengths) #Lowest number of voltages for a working device
    all_pixel_summary_array = np.empty((min_length,len(RESULT_CATEGORIES),len(working_pixels)))
    for i in range (0, len(working_pixels)):
        pixel_code = working_pixels[i]
        pixel_dat = all_pixel_summary_data[i] #Issues here 2023-11-21
        all_pixel_summary_array[:,:,i]=pixel_dat[0:min_length]

    average_summary = np.average(all_pixel_summary_array,axis=2)
    stdev_summary = np.std(all_pixel_summary_array, axis=2)

    average_spectra = np.average(all_pixel_spectral_data, axis=0)
    std_spectra = np.std(all_pixel_spectral_data, axis=0)
    create_smeared_graph(WAVELENGTHS, average_spectra, std_spectra,local_file_save_path,device_id,"Wavelength (nm)", "Intensity (Counts)", 
        "Average Emission Spectra at L = "+str(luminance_threshold)+" cd/m2", "average_spectra", 'linear', 'linear', 380, 0, median_color, 'smeared', None)

    spectral_avg_df = pd.DataFrame()
    spectral_avg_df["Wavelength (nm)"] = WAVELENGTHS
    spectral_avg_df["Intensity (Counts)"] = average_spectra
    spectral_avg_df["Standard Deviation (Counts)"] = std_spectra
    spectral_avg_df.to_csv(local_file_save_path+"/Raw Data/"+device_id+"_avgspectra_"+str(luminance_threshold)+"cdm2.csv", sep='\t', encoding='utf-8')

    optoelectronic_avg_df = pd.DataFrame(columns=[RESULT_CATEGORIES[0]])
    optoelectronic_avg_df[RESULT_CATEGORIES[0]]=average_summary[:, 0]
    for i in range (1, len(RESULT_CATEGORIES)):
        optoelectronic_avg_df[RESULT_CATEGORIES[i]]=average_summary[:, i]
        optoelectronic_avg_df[RESULT_CATEGORIES[i]+" stdev"]=stdev_summary[:, i]
    optoelectronic_avg_df.to_csv(local_file_save_path+"/Raw Data/"+device_id+"_avgoutput.csv", sep='\t', encoding='utf-8')

    von_index = next(x[0] for x in enumerate(average_summary[:, 0]) if x[1] > on_voltage) #turnon index
    for pairing in DESIRED_PAIRINGS:
        pairing_xindex = pairing[0]
        pairing_yindex = pairing[1]
        use_cutoff = pairing[2]

        min_index = 0 #This algorithm is not foolproof
        if use_cutoff:
            max_value = np.amax(average_summary[von_index:,pairing_yindex])
            for i in range (0, len(stdev_summary[:,pairing_yindex])):
                if max_value-stdev_summary[i,pairing_yindex] > 0 and max_value-stdev_summary[i+1,pairing_yindex] > 0:
                    min_index = i
                    break
        
        x = average_summary[min_index:, pairing_xindex]
        y = average_summary[min_index:, pairing_yindex]

        x_label = RESULT_CATEGORIES[pairing_xindex]
        y_label = RESULT_CATEGORIES[pairing_yindex]

        cutoff_voltage = average_summary[min_index, 0]
        valid_rows = all_pixel_pd[all_pixel_pd['Voltage(V)']>=cutoff_voltage]
        x_s = valid_rows[x_label].values
        y_s = valid_rows[y_label].values
        y_var_s = 0

        y_var = stdev_summary[min_index:, pairing_yindex]
        title = x_label+' vs. '+y_label 
        file_label = x_label.split('(')[0] + '_vs_' + y_label.split('(')[0]

        x_scale = RESULT_SCALE[pairing_xindex]
        y_scale = RESULT_SCALE[pairing_yindex]

        x_min = MIN_VALUE[pairing_xindex]
        y_min = MIN_VALUE[pairing_yindex]
  
        x_min = max(x_min, np.amin(x))
        y_max = create_smeared_graph(x,y,y_var,local_graph_save_path,device_id,x_label,y_label,title,file_label,x_scale,y_scale,x_min,y_min, median_color, 'smeared', None)
        create_smeared_graph(x_s,y_s,y_var_s,local_graph_save_path,device_id,x_label,y_label,title,file_label+"_scatter",x_scale,y_scale,x_min,y_min, median_color, 'scatter', y_max)

def create_smeared_graph(x, y, y_var, local_file_save_path, device_id, x_label, y_label, title, file_label, x_scale,y_scale,x_min,y_min, median_color,graph_type,y_max):
    fig, ax = plt.subplots()

    if graph_type=='smeared':
        ax.plot(x, y, color=median_color)
        ax.fill_between(x, y+y_var,y-y_var,alpha=0.3, color=median_color)
    elif graph_type=='scatter':
        ax.scatter(x,y,color=median_color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.yscale(y_scale)
    plt.xscale(x_scale)
    ax.set_xlim([x_min,None])
    ax.set_ylim([y_min,y_max])

    y_max = ax.get_ylim()[1]

    plt.savefig(local_file_save_path+"/"+device_id+'_' + file_label+'.png')
    plt.clf()
    plt.close('all')
    return y_max

#Creates updated data for fully processed OLED runs for the online spreadsheet
def update_oled_output_data(oled_data, device_id, regression_row):

    working_devices = oled_data[oled_data['Status'].str.contains('Working')]

    DEFAULT_COMPARISON_LUMINANCE = [250, 100] 
    luminance_threshold = 0
    for default_lum in DEFAULT_COMPARISON_LUMINANCE:
        luminance_above_threshold_list = working_devices['Max Luminance (cd/m2)'].values > default_lum
        num_above = sum(bool(x) for x in luminance_above_threshold_list) 
        if num_above > len(working_devices)*0.5:
            luminance_threshold = default_lum
            break

    working_devices = working_devices[working_devices['Max Luminance (cd/m2)'] > luminance_threshold]
    num_working = len(working_devices)
    num_shorted = len(oled_data[oled_data['Status'].str.contains('Short Circuit')])

    CATEGORIES_TO_FIND = ['Max Luminance (cd/m2)', 'Turnon Voltage (V)', 'CE(cd/A)_max','CE(cd/A)_avg', 'EQE(%)_avg', 'PE(lm/W)_avg',
        "\u03BBmax (nm) @ "+str(luminance_threshold)+" cd/m2", 'FWHM (nm) @ '+str(luminance_threshold)+' cd/m2']

    categories = []
    for item in CATEGORIES_TO_FIND:
        if item == 'CE(cd/A)_max':
            categories.append(max(working_devices[item].values))
        categories.append(np.median(working_devices[item].values))
        if item == 'CE(cd/A)_avg':
            categories.append(np.std(working_devices[item].values))
            #categories.append(max(working_devices[item].values))
           

    COLORS_TO_FIND = ["CIE-x @ "+str(luminance_threshold)+" cd/m2", "CIE-y @ "+str(luminance_threshold)+" cd/m2"]
    median_ciex = np.median(working_devices[COLORS_TO_FIND[0]].values)
    median_ciey = np.median(working_devices[COLORS_TO_FIND[1]].values)
    median_color = get_hexcolor_from_cie(median_ciex, median_ciey)
    categories.append(median_color)

    categories.append(regression_row['Material'].values[0])
    categories.append(str(regression_row['Max Luminance (cd/m2)'].values[0])[0:5])

    return [device_id, 'Processed', num_working, num_shorted] + categories, median_color

#Filters the OLED data to look for missing data and pick the 'best' duplicate instead of using redundant data
def filter_compressed_data(compressed_data):
    filtered_data = pd.DataFrame(columns=compressed_data.columns.values)
    for pixel_name in OLED_PIXEL_NAMES:
        pixel_rows = compressed_data[compressed_data['Pixel'].str.contains(pixel_name)]
        #print(pixel_rows)
        if len(pixel_rows.index)==1:
            filtered_data.loc[len(filtered_data)] = pixel_rows.values[0]
        elif len(pixel_rows.index)==0:
            filtered_data.loc[len(filtered_data)] = [pixel_name, '-', '-', '-', 'Missing Data'] + ['-']*(len(compressed_data.columns.values)-5)
        else:
            working_rows = pixel_rows[pixel_rows['Status']=='Working']
            best_device_row = pixel_rows[pixel_rows['Max Luminance (cd/m2)'] == pixel_rows['Max Luminance (cd/m2)'].max()]
            filtered_data.loc[len(filtered_data)] = best_device_row.values[0]
    return filtered_data

#Take in the raw OLED data, return compressed useful data for further processing and to save... Let's update this to be better
def create_compressed_oled_data(reference_folder, data_file_names):
    
    TURNON_LUMINANCE_THRESHOLD = 2

    columns_output=["Pixel", "Max Current (A)", "Max Luminance (cd/m2)", "Status", "Turnon Voltage (V)"]

    #Add in the max/avg columns... We will need to set a threshold for this! eg 10 cd/m2
    SUB_CATEGORY = ['max', 'avg']
    CATEGORIES_A = ["EQE(%)", "CE(cd/A)", "PE(lm/W)"]
    for i in range (0, len(CATEGORIES_A)):
        for j in range (0, len(SUB_CATEGORY)):
            columns_output.append(CATEGORIES_A[i]+"_"+SUB_CATEGORY[j])

    L_THRESHOLDS = [100, 250] #Luminances we want to assess performance at, and the first one is turn-on luminance 
    CATEGORIES = ["V (V)", "EQE(%)", "CE(cd/A)", "PE(lm/W)", "\u03BBmax (nm)", "FWHM (nm)", "CIE-x", "CIE-y", "Hex Color"] #Categories we are analyzing at each L_THRESHOLD except the first

    graph_labels = []
    for i in range (0, len(L_THRESHOLDS)):
        for j in range (0, len(CATEGORIES)):
            threshold_lum = str(L_THRESHOLDS[i])
            columns_output.append(CATEGORIES[j] + " @ " + threshold_lum + " cd/m2")

    #Create a dataframe to store the data... make this more automatic based on L_THREHOLDS
    output_df = pd.DataFrame(columns=columns_output)

    for data_file_name in data_file_names: #Go through the files
        input_df = pd.read_csv(reference_folder+'/'+data_file_name, delimiter='\t')
        input_df['CE(cd/A)']=input_df['L(cd/m2)']/input_df['J(mA/cm2)']/10 #Calculating the current efficiency!
        
        pixel = data_file_name.split('.')[0]
        current_max = np.amax(input_df['Current(A)'].values)
        luminance_max = round(np.amax(input_df['L(cd/m2)'].values)*100)/100

        status = 'Working'
        OPEN_CIRCUIT_THRESHOLD = 1E-5 #We are open if we don't exceed this current (ie no connection or very high resistance)
        SHORT_CIRCUIT_MAX_LUMINANCE = 50 #If we shorted, we won't exceed this luminance
        LOW_LUMINANCE_THRESHOLD = 100
        if current_max < OPEN_CIRCUIT_THRESHOLD:
            status = 'Open Circuit'
        elif luminance_max < SHORT_CIRCUIT_MAX_LUMINANCE and current_max > 0.01:
            status = 'Short Circuit'
        elif luminance_max < LOW_LUMINANCE_THRESHOLD:
            status = 'Low Luminance'

        if status == 'Working':
            try:
                spectral_df = pd.read_csv(reference_folder+'/'+pixel+"_counts.txt", delimiter='\t')  
            except:
                status = "Missing Data"

        turnon_voltage = '-' #Default if not working
        summary_characteristics = ['-']*(len(SUB_CATEGORY)*len(CATEGORIES_A)+len(CATEGORIES)*len(L_THRESHOLDS))

        if status == 'Working': #Only further analyze working devices
            luminances = input_df['L(cd/m2)'].values
            voltages = input_df['Voltage(V)'].values

            index_turnon = next(x[0] for x in enumerate(luminances) if x[1] > TURNON_LUMINANCE_THRESHOLD)
            turnon_voltage = round(voltages[index_turnon] * 10)/10

            reduced_df = input_df[input_df['L(cd/m2)'] > TURNON_LUMINANCE_THRESHOLD] #Only include data points above a certain luminance
            summary_characteristics = []
            for i in range (0, len(CATEGORIES_A)):
                for j in range (0, len(SUB_CATEGORY)):
                    category = CATEGORIES_A[i]
                    type_cat = SUB_CATEGORY[j]

                    if type_cat == 'avg':
                        summary_characteristics.append(np.average(reduced_df[category].values))
                    elif type_cat == 'max':
                        summary_characteristics.append(max(reduced_df[category].values))

            spectral_wavelengths = np.arange(380, 780, 0.5) #These are set in the program
            low_luminance = False
            spectra_list = []

            for threshold_lum in L_THRESHOLDS: #Find the voltage where certain luminances are exceeded
                try: 
                    index_threshold = next(x[0] for x in enumerate(luminances) if x[1] > threshold_lum) #Let's work on a different way to do this
                    summary_characteristics.append(input_df['Voltage(V)'].values[index_threshold])
                except:
                    low_luminance = True
                    for i in range (0, len(CATEGORIES)):
                        summary_characteristics.append("-") #Just add blank data if we can't reach this luminance
                try:
                    if low_luminance == False:
                        spectra = spectral_df.iloc[:, index_threshold]
                        smoothed_spectra = savgol_filter(spectra, 51, 3) #This smoothes the spectra
                        spectra_list.append(smoothed_spectra)
                        spectral_peak_index = np.argmax(smoothed_spectra)
                        if np.amax (smoothed_spectra) > 0:
                           
                            norm_spectra = smoothed_spectra / np.amax (smoothed_spectra)

                            index_rise = next(x[0] for x in enumerate(norm_spectra[0:np.argmax(norm_spectra)]) if x[1] > 0.5)
                            index_fall = next(x[0] for x in enumerate(norm_spectra[np.argmax(norm_spectra):]) if x[1] < 0.5)+np.argmax(norm_spectra)

                            summary_characteristics.append(input_df['EQE(%)'].values[index_threshold])
                            summary_characteristics.append(input_df['CE(cd/A)'].values[index_threshold])
                            summary_characteristics.append(input_df['PE(lm/W)'].values[index_threshold])
                            summary_characteristics.append(spectral_wavelengths[spectral_peak_index])
                            summary_characteristics.append(spectral_wavelengths[index_fall]-spectral_wavelengths[index_rise])

                            cie_x,cie_y = get_cie_coordinates_from_spectra(smoothed_spectra, spectral_wavelengths)
                            summary_characteristics.append(cie_x)
                            summary_characteristics.append(cie_y)
                            color = get_hexcolor_from_cie(cie_x, cie_y)
                            summary_characteristics.append(color)
                        else:
                            if low_luminance == False:
                                raise Exception("issue with spectral output")
                except Exception as e:
                    print(e)
                    for i in range (0, len(CATEGORIES)-1):
                        summary_characteristics.append("-")
        #print(columns_output)
        #print(len(columns_output))
        data = [pixel, current_max, luminance_max, status, turnon_voltage] + summary_characteristics
        #print(data)
        #print(len(data))
        output_df.loc[len(output_df)] = data
    return output_df

#Get a color that can be plotted from cie coordinates
def get_hexcolor_from_cie(cie_x, cie_y):
    REFERENCE_MATRIX = np.array([[0.64, 0.33, 0.03],
                                    [0.30, 0.60, 0.1],
                                     [0.15, 0.06, 0.79]])
    color_vector_cie = np.array([cie_x, cie_y, 1-cie_x-cie_y])
    rgba_vector = np.matmul(np.linalg.inv(REFERENCE_MATRIX), color_vector_cie)
    if np.any(rgba_vector < 0):
        rgba_vector += -np.amin(rgba_vector)
    if not np.all(rgba_vector==0):
            # Normalize the rgb vector
            rgba_vector /= np.max(rgba_vector)
    hex_rgb = (255 * rgba_vector).astype(int)

    return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

#Convert spectral data into CIE coordinates
def get_cie_coordinates_from_spectra(spectra, wavelengths):
    reference_spectra_df = np.loadtxt('cie_reference.txt', delimiter=" ", dtype=float)

    reference_spectra_x_inter = []
    reference_spectra_y_inter = []
    reference_spectra_z_inter = []
    reference_wavelengths = reference_spectra_df[:, 0]
    reference_x = reference_spectra_df[:, 1]
    reference_y = reference_spectra_df[:, 2]
    reference_z = reference_spectra_df[:, 3]

    for i in range (0, len(wavelengths)):
        wavelength = wavelengths[i]
        reference_index = next(x[0] for x in enumerate(reference_wavelengths) if x[1] >= wavelength)
        reference_wavelength = reference_wavelengths[reference_index-1]
        delta_wavelength = wavelength - reference_wavelength

        if delta_wavelength > 0:
            interpolated_x = delta_wavelength / 5 * (reference_x[reference_index]-reference_x[reference_index-1])+reference_x[reference_index-1]
            interpolated_y = delta_wavelength / 5 * (reference_y[reference_index]-reference_y[reference_index-1])+reference_y[reference_index-1]
            interpolated_z = delta_wavelength / 5 * (reference_z[reference_index]-reference_z[reference_index-1])+reference_z[reference_index-1]
        else:
            interpolated_x = reference_x[reference_index]
            interpolated_y = reference_y[reference_index]
            interpolated_z = reference_z[reference_index]

        reference_spectra_x_inter.append(interpolated_x)
        reference_spectra_y_inter.append(interpolated_y)
        reference_spectra_z_inter.append(interpolated_z)

    X = np.dot(reference_spectra_x_inter, spectra)
    Y = np.dot(reference_spectra_y_inter, spectra)
    Z = np.dot(reference_spectra_z_inter, spectra)

    return X/(X+Y+Z), Y/(X+Y+Z)

#Create and save a diagram of the OLED stack
def generate_oled_stack_png(device_id, material_data_df, deposition_output_df, save_folder):

    deposition_runs_df = deposition_output_df[deposition_output_df['Deposition ID'].str.contains(device_id)]

    #Organic: [Name, type, subtype, thickness, color, HOMO, LUMO, T1, wf] 
    #Subtype = EMH, EMFD, EMTD, None

    material_properties_list = []
    material_properties_list.append(["ITO", "Metal", None, 1000, "#000000", 0, 0, -4.7]) #We always use ITO

    num_em_materials = 0
    for deposition_index,deposition_row in deposition_runs_df.iterrows(): #Fill up the material properties list
        material_name = deposition_row['Material']
        material_thickness = deposition_row['Thickness Deposited on Substrate (A)']
        material_subtype = deposition_row['Deposition Subtype']
        if material_subtype in ["EMH", "EMTD", "EMFD"]:
            num_em_materials+=1
       
        material_energy_row = material_data_df[material_data_df['Material']==material_name]
        material_type = material_energy_row['Material Type'].values[0]
        material_homo = material_energy_row['HOMO (eV)'].values[0]
        material_lumo = material_energy_row['LUMO (eV)'].values[0]
        material_workfunction = material_energy_row['Work Function (eV)'].values[0]
        material_color = material_energy_row['Color'].values[0]

        entry = [material_name, material_type, material_subtype, material_thickness, material_color, material_homo, material_lumo, material_workfunction]
        material_properties_list.append(entry)

    energies = np.array(material_properties_list)[:, 5:].astype(float).flatten()
    energies = np.delete(energies, np.where(energies == 0))
    min_energy = np.amin(energies)
    max_energy = np.amax(energies)

    #Take the material property list and generate a graph!
    fig, ax = plt.subplots(figsize=(6.4*len(deposition_runs_df)/4,4.8))
    count = 0
    dist = 0.25
    em_count=0
    metal_count=0
    extra_space = True
    emh_lumo = 0
    
    for material in material_properties_list:
        type = material[1]
        subtype = material[2]
        name = material[0]
        thickness = material[3]
        color_mat = material[4]
        text_offset = -0.22

        if (subtype in ["EMH", "EMTD", "EMFD"]):
            em_count=em_count+1
        if type=="Metal" or type=="Interface":
            width = 1.75

            wf = float(material[7])

            p1 = [dist,wf]
            p2 = [dist+width,wf]
            xs = [p1[0],p2[0]]
            ys = [p1[1],p2[1]]

            ax.plot(xs, ys, color=color_mat)
            ax.text(p1[0], p1[1]+0.10, name+" ("+str(thickness)+ " " + r'$\AA$' + ")",fontsize="small")
            dist=dist+width+0.25
            if type=="Metal":
                ax.text(p1[0], p1[1]+text_offset,str(wf)[0:5]+" eV",fontsize="x-small")
                metal_count=metal_count+1
        elif type=="Organic":
            try:
                homo = float(material[5])
                lumo = float(material[6])
            except:
                print("HOMO/LUMO values could not be parsed")
                homo = 0
                lumo = 0

            #Adjust this to make up for when the LUMO Is too close for the EMTD
            name_offset = lumo+0.1
            if emh_lumo < 0 and emh_lumo - lumo < 0.5 and (subtype == "EMTD" or subtype == "EMFD"):
                name_offset = homo-0.30

            ax.text(dist,name_offset,name[0:11]+" ("+str(thickness)[0:4]+ " " + r'$\AA$' + ")",fontsize="small")
            
            ax.text(dist+0.05,lumo+text_offset,str(lumo)[0:5]+" eV",fontsize="x-small")
            if subtype == "EMFD" or subtype == "EMTD":
                text_offset=0.05
            ax.text(dist+0.05,homo+text_offset,str(homo)[0:5]+" eV",fontsize="x-small")
            if subtype != "EMH":
                ax.add_patch(Rectangle((dist,homo), 1.75, lumo-homo, color=color_mat))
                dist=dist+2
            elif subtype == "EMH":
                ax.add_patch(Rectangle((dist,homo), 2*(num_em_materials-1)+0.25, lumo-homo, linestyle = 'dashed', clip_on=False, edgecolor=color_mat, facecolor='None'))
                dist=dist+0.25
                emh_lumo = lumo
            count=count+1
        if em_count==num_em_materials and num_em_materials>0 and extra_space:
            dist=dist+0.25
            extra_space = False
    ax.set_ylim(min_energy-1,max_energy+1)
    ax.set_xlim(0,dist)
    plt.ylabel("Energy (eV)")

    title = device_id + " Energy Level Diagram"

    plt.title(title)
    ax.axes.xaxis.set_visible(False)

    plt.savefig(save_folder+'/'+title)
    plt.clf()

    return None

#This looks for simple correlations... how does changing one thickness affect output?
def check_correlations(working_devices,stack_reports,device_id, local_file_save_path):

    material_list = stack_reports.columns.values[1:]

    #Step 1: Generate X matrix from the stack_reports!
    print("Detecting Correlations...")
    X_matrix = np.array(stack_reports.iloc[:, 1:].values)
   # print ("X_matrix shape: ", X_matrix.shape, " (Should be 20xNumber of Layers+1)")
    
   #Create a directory if it doesn't exist
    local_corr_save_path = local_file_save_path + '/Correlations'
    if path.exists(local_corr_save_path):
        print("Local output directory already exists (Correlations)...")
    else:
        print("Creating Local Output Directory (Correlations): " +device_id)
        os.mkdir(local_corr_save_path)

     #This code section is to allow calculation even if we don't meet the threshold
    DEFAULT_COMPARISON_LUMINANCE = [250, 100] 
    luminance_threshold = 0
    for default_lum in DEFAULT_COMPARISON_LUMINANCE:
        luminance_above_threshold_list = working_devices['Max Luminance (cd/m2)'].values > default_lum
        num_above = sum(bool(x) for x in luminance_above_threshold_list) 
        if num_above > len(working_devices)*0.5:
            luminance_threshold = default_lum
            break
    desired_outputs = ['Max Luminance (cd/m2)', 'EQE(%)_max', 'EQE(%)_avg', 'PE(lm/W)_max', 'PE(lm/W)_avg', 'CE(cd/A)_max', 'CE(cd/A)_avg']

    working_devices = working_devices[working_devices['Max Luminance (cd/m2)'] > luminance_threshold]
    #print(working_devices)

    #Step 2: Get Y matrix from the oled output data!
    Y = np.array(working_devices.loc[:,desired_outputs].values).astype(float)
   # print ("Y_matrix shape: ", Y.shape, " (Should be number of working pixels x number of outputs)")
    #print(Y)

    X = []
    for index, row in working_devices.iterrows():
        if row['Max Luminance (cd/m2)'] > luminance_threshold:
            X.append(list(X_matrix[index]))
    X = np.array(X).astype(float)
    #print("Updated X_matrix shape: ", X.shape, " (Should be number of working pixels x number of inputs+1)")
    #print(X)

    correlation_df = pd.DataFrame()
    correlation_df['Material']=material_list

    y_count = 0
    for y in np.transpose(Y):
        y_name = desired_outputs[y_count]
        material_count = 0
        rs = []

        for material in material_list:
            x = X[:, material_count]
            r = np.corrcoef(x, y)[0,1]
            rs.append(r)
            #print("Material: " + material + ", Correlation: " + str(r)[0:5])
            material_count+=1

        correlation_df[y_name]=rs
        best_correlation_index = np.argmax(np.abs(rs))
        best_material = material_list[best_correlation_index]
        #print("Material most correlated to " + y_name + ": " + best_material + ", r=" + str(rs[best_correlation_index])[0:5])
        
        fig, ax = plt.subplots()
        plt.scatter(X[:, best_correlation_index], y)
        plt.title("Top Correlation for: " + y_name + " r = " + str(rs[best_correlation_index])[0:5])
        ax.set_xlabel(best_material + " Thickness (" + r'$\AA$' + ')')
        ax.set_ylabel(y_name)
        plt.savefig(local_corr_save_path+"/"+y_name.split('(')[0]+'_top_correlation.png')
        plt.clf()

        y_count+=1
    correlation_df.to_csv(local_corr_save_path+"/"+device_id+"_correlations.csv", sep='\t', encoding='utf-8')

    return correlation_df[np.abs(correlation_df['Max Luminance (cd/m2)']) == np.abs(correlation_df['Max Luminance (cd/m2)']).max()]

#The following 5 functions are used for regression but it all seems a bit... overly complex and not useful at least for 1 device
def create_regression_report(oled_data, stack_reports, device_id, local_file_save_path):
    
    pixel_array = ['A1', 'B1', 'C1', 'D1', 'A2', 'B2', 'C2', 'D2', 'A3', 'B3', 'C3', 'D3', 'A4', 'B4', 'C4', 'D4', 'A5', 'B5', 'C5', 'D5']

    material_list = stack_reports.columns.values[1:]
    print(material_list)

    #Step 1: Generate X matrix from the stack_reports!
    print("Generating Regression Report")
    X_matrix = np.array(stack_reports.iloc[:, 1:].values)
    X_matrix = np.c_[np.ones(X_matrix.shape[0]),  X_matrix] #Add constant term 
    print ("X_matrix shape: ", X_matrix.shape, " (Should be 20xNumber of Layers+1)")

    #Step 2: Get Y matrix from the oled output data!
    desired_y_column_names = ['Max Luminance (cd/m2)', 'EQE(%) @ 250 cd/m2'] #Choose the column headers you want to analyze...
    working_devices = oled_data[oled_data['Status'].str.contains('Working')]
    Y = np.array(working_devices.loc[:,desired_y_column_names].values).astype(float)
    print ("Y_matrix shape: ", Y.shape, " (Should be number of working pixels x number of outputs)")

    X = []
    for index, row in oled_data.iterrows():
        if row['Status'] == 'Working':
            X.append(list(X_matrix[index]))
    X = np.array(X).astype(float)
    print("Updated X_matrix shape: ", X.shape, " (Should be number of working pixels x number of inputs+1)")

    y_count=0
    r_squared_matrix = []
    f_val_matrix = []
    r_adj_matrix = []
    beta_matrix = []
    T_matrix = []
    for y_col in np.transpose(Y): #y-data set
        y_name = desired_y_column_names[y_count]
        print ("Y-variable: ", y_name)
        y_col = np.transpose(y_col)
        x_temp = X
        mat_temp = material_list
        r_sq_y = []
        f_val_y = []
        r_adj_y = []
        beta_matrix_y = []
        T_matrix_y = []
        for i in range (0, X.shape[1]-1): #for every x-variable
            BETA,r_squared,f_val,T,r_adj = regression_analysis(x_temp,y_col)
            r_sq_y.append(r_squared)
            f_val_y.append(f_val)
            r_adj_y.append(r_adj)
            beta_matrix_y.append(fill_in_zeros(BETA, material_list, mat_temp))
            T_matrix_y.append(fill_in_zeros(T,material_list, mat_temp))
            x_temp,mat_temp = remove_lowest_item(x_temp,mat_temp,T)
        y_count=y_count+1
        r_squared_matrix.append(r_sq_y)
        r_adj_matrix.append(r_adj_y)
        f_val_matrix.append(f_val_y)
        beta_matrix.append(beta_matrix_y)
        T_matrix.append(T_matrix_y)
        #print ("Adjusted R2: ", y_name, "\n", r_adj_y)

    regression_reports = [] #Separately create the documents
    for i in range (0, y_count):
        t_vals = T_matrix[i]
        r_adj_vals = r_adj_matrix[i]
        y_name = desired_y_column_names[i]
        beta_vals = beta_matrix[i]
        r2_vals = r_squared_matrix[i]

        sheet = []
        entry = np.insert(material_list, 0, "Material")
        entry = np.insert(entry, len(entry), ["R2-Adj", "R2", "Best Model (?)"]) #Not sure if this works
        sheet.append(list(entry))

        for iteration_number in range (0, len(beta_vals)):
            best_model = "No"
            if r_adj_vals[iteration_number]==np.max(r_adj_vals): #Is this our best model?
                best_model="Yes"
            header = "Beta_"+str(iteration_number)
            entry = np.insert(np.array(beta_vals[iteration_number]).astype(str), 0, header)
            entry = np.insert(entry, len(entry), [r_adj_vals[iteration_number], r2_vals[iteration_number], best_model])
            sheet.append(list(entry))
            header = "T_"+str(iteration_number)
            entry = np.insert(np.array(t_vals[iteration_number]).astype(str), 0, header)
            entry = np.insert(entry, len(entry), ["-", "-", best_model])
            sheet.append(list(entry))

            #add in whether or not the t-vales are significant into a column
            dof = X.shape[0]-(X.shape[1]-iteration_number) #degrees of freedom for T-calculation
            t_val_sig = stats.t.ppf(1-0.025, dof) #get the t_value for this iteration
            #print(t_val_sig)
            yn_column = []
            for t_val in t_vals[iteration_number]:
                sig_str = "N"
                if np.absolute(t_val) > t_val_sig:
                    sig_str = "Y"
                if t_val==0:
                    sig_str = "0"
                yn_column.append(sig_str)
            header="Sig?"
            entry = np.insert(header, 1, np.array(yn_column).astype(str))
            entry = np.insert(entry, len(entry), ["-", "-", best_model])
            sheet.append(list(entry))

        sheet = np.transpose(sheet)
        report = pd.DataFrame(sheet[1:], columns=sheet[0])
        report.to_csv(local_file_save_path+"/"+y_name.split('(')[0]+"_regression.csv", sep='\t', encoding='utf-8')


    return regression_reports, desired_y_column_names, X, material_list, Y #For program to use in spreadsheet

def fill_in_zeros(matrix,mat_list_original,mat_list_reduced):
    matrix_filled = []
    for i in range (0, len(mat_list_original)):
        matched = False
        for j in range (0, len(mat_list_reduced)):
            if mat_list_original[i]==mat_list_reduced[j]:
                matched=True
                matrix_filled.append(matrix[j])
        if matched==False:
            matrix_filled.append(0)
    return matrix_filled

def remove_lowest_item(X,material_list,T):
    min_t_index = np.argmin(np.absolute(T[1:]))
    #print("Least significant item: ", material_list[min_t_index])
    X = np.delete(X,min_t_index,1)
    material_list = np.delete(material_list,min_t_index,0)
    return X,material_list

def remove_items_below_threshold(X,material_list,T,threshold):
    remove = []
    for i in range (1, X.shape[1]): #For each parameter
        if any(np.absolute(T[i])>threshold)==False:
            remove.append(i)
    X_red = np.delete(X,remove,1)
    material_list_red = np.delete(material_list,remove,0)

    return X_red, material_list_red

def regression_analysis(X,Y):
    #print("Attempting regression...")
    A = np.linalg.pinv(np.matmul(np.transpose(X), X)) #(X_T*X)^-1
    B = np.matmul(np.transpose(X), Y) #(X_T*Y)
    BETA = np.matmul(A,B)
    y_hat = np.matmul(X, BETA)

    SSE = np.sum(np.power(Y - y_hat, 2),0)
    SST = np.sum(np.power(Y - np.average(Y,0), 2),0)
    SSM = np.sum(np.power(y_hat - np.average(Y,0),2),0)
    DFM = X.shape[1] - 1 #p - 1
    DFE = X.shape[0] - X.shape[1] #n - p
    DFT = X.shape[0] - 1 #n - 1
    f_val = (SSM/DFM) / (SSE/DFE)
    r_squared = 1 - SSE/SST
    r_adj = 1 - (1-r_squared)*DFT/DFE

    #SSE = np.reshape(SSE,[len(SSE),1,1])
    #A = np.reshape(A, [1,A.shape[0],A.shape[1]])
    covX = SSE/DFE*A
    diagonals = np.transpose(np.diagonal(covX))

    T = BETA / np.sqrt(diagonals) #Significance matrix
    return BETA,r_squared,f_val,T,r_adj

#Utility function for plotting lines from points
def get_line_ends(p1, p2):
    return [ [p1[0], p2[0]], [p1[1], p2[1]] ]

#Create CIE plot for each non-redundant pixel
def create_CIE_scatter_plot(working_devices, run_id, local_file_save_path):

    print("Creating CIE scatter plot...")
    fig, ax = plt.subplots()

    #BT2020
    red_ref = [0.708, 0.292]
    blue_ref = [0.131, 0.046]
    green_ref = [0.292, 0.797]

    RB_line_ends=get_line_ends(red_ref, blue_ref)
    RG_line_ends=get_line_ends(green_ref, red_ref)
    GB_line_ends= get_line_ends(blue_ref, green_ref)

    plt.plot(RB_line_ends[0], RB_line_ends[1], '--', color='black')
    plt.plot(RG_line_ends[0], RG_line_ends[1], '--', color='black')
    plt.plot(GB_line_ends[0], GB_line_ends[1], '--', color='black')

    plt.plot(red_ref[0], red_ref[1], 'o', color=get_hexcolor_from_cie(red_ref[0], red_ref[1]))
    plt.plot(blue_ref[0], blue_ref[1], 'o', color=get_hexcolor_from_cie(blue_ref[0], blue_ref[1]))
    plt.plot(green_ref[0], green_ref[1], 'o', color=get_hexcolor_from_cie(green_ref[0], green_ref[1]))

    DEFAULT_COMPARISON_LUMINANCE = [250, 100] 
    LUMINANCE_CRITERIA = 0
    for default_lum in DEFAULT_COMPARISON_LUMINANCE:
        luminance_above_threshold_list = working_devices['Max Luminance (cd/m2)'].values > default_lum
        num_above = sum(bool(x) for x in luminance_above_threshold_list) 
        if num_above > len(working_devices)*0.5:
            LUMINANCE_CRITERIA = default_lum
            break

    #Update this to use pandas... Get color for each point...
    for index,row in working_devices.iterrows():
        try:
            plt.plot(float(row['CIE-x @ '+ str(LUMINANCE_CRITERIA) + ' cd/m2']), float(row['CIE-y @ '+ str(LUMINANCE_CRITERIA) + ' cd/m2']), 'o', color=row['Hex Color @ '+ str(LUMINANCE_CRITERIA) + ' cd/m2'])
        except:
            None

    cie_file_name = run_id + " CIE Plot.png"
    plt.title("CIE Plot: " + run_id + " @ L = "+ str(LUMINANCE_CRITERIA) + ' cd/m2')
    plt.savefig(local_file_save_path+"/"+cie_file_name)
    plt.clf()
    return cie_file_name

#Summarize stack architecture including calculating corrected thicknesses
def create_stack_report(deposition_output_df, tooling_df, device_id, local_file_save_path):
    deposition_runs_df = deposition_output_df[deposition_output_df['Deposition ID'].str.contains(device_id)]
    pixel_locations = pd.read_csv('OLED_pixel_positions.txt', delimiter='\t')

    material_list = list(deposition_runs_df['Material'].values)
    stack_df = pd.DataFrame(columns=(['Pixel'] + material_list))

    M = np.empty((len(material_list),4), dtype=float)
    for i in range (0, len(material_list)):
        material = material_list[i]
        pixel_abct = tooling_df[tooling_df['Material']==material]
        a = float(pixel_abct['Tooling (A)'].values[0])
        b = float(pixel_abct['Tooling (B)'].values[0])
        c = float(pixel_abct['Tooling (C)'].values[0])
        t_centroid = float(pixel_abct['Tooling (Centroid)'].values[0])
        t_ic6 =  float(deposition_runs_df[deposition_runs_df['Material']==material]['Thickness Deposited on Substrate (A)'].values[0])
        M[i]=[a,b,c,t_ic6/t_centroid]

    for pixel in OLED_PIXEL_NAMES:
        pixel_xy = pixel_locations[pixel_locations['pixel']==pixel]
        x = pixel_xy['x'].values[0]
        y = pixel_xy['y'].values[0]

        stack_df.loc[len(stack_df)]=pixel
        #print(stack_df)
        for i in range (0, len(material_list)):
            material = material_list[i]
            stack_df.at[len(stack_df)-1,material]=(M[i,0]*x+M[i,1]*y+M[i,2])*M[i,3]


    stack_df.to_csv(local_file_save_path+"/Raw Data/"+device_id+"_stack_details.csv", sep='\t', encoding='utf-8')        
    return stack_df

#Generate the data for a visual report
def create_visual_report(oled_data, run_id, local_file_save_path):
    RECT_WIDTH = 0.5
    RECT_HEIGHT = 0.5
    LUMINANCE_CRITERIA = 250

    working_devices = oled_data[oled_data['Status'].str.contains('Working')]
    DEFAULT_COMPARISON_LUMINANCE = [250, 100] 
    for default_lum in DEFAULT_COMPARISON_LUMINANCE:
        luminance_above_threshold_list = working_devices['Max Luminance (cd/m2)'].values > default_lum
        num_above = sum(bool(x) for x in luminance_above_threshold_list) 
        if num_above > len(working_devices)*0.5:
            LUMINANCE_CRITERIA = default_lum
            break
    max_lum = max(np.array(working_devices['Max Luminance (cd/m2)'].values).astype('float'))

    print("Creating visual report...")

    fig, ax = plt.subplots()
    count=0
    for i in range (0, 4):
        for j in range (0, 5):
            pixel_name = OLED_PIXEL_NAMES[count] #A1 all the way to D5
            pixel_row = oled_data[oled_data['Pixel'].str.contains(pixel_name)]
            pixel_status = pixel_row['Status'].values[0] 

            if pixel_status == 'Working':
                color = pixel_row['Hex Color @ '+ str(LUMINANCE_CRITERIA) + ' cd/m2'].values[0]

                CE = pixel_row['CE(cd/A)_avg'].values[0]
                L_MAX = pixel_row['Max Luminance (cd/m2)'].values[0]
                TURNON_V = pixel_row['Turnon Voltage (V)'].values[0]
                EQE_string = "CE(Avg): " + str(CE)[0:4]+ " cd/A"
                L_MAX_string = "Max Luminance:\n" + str(int(L_MAX))+ ' cd/m2'
                turnon_string = "Turnon: " + str(TURNON_V) + " V"

                full_string = turnon_string + '\n' + EQE_string + '\n' + L_MAX_string

                color_alpha = float(L_MAX)/max_lum

                if color == '-':
                    color = 'Brown'

                ax.text((i+0.5)*RECT_WIDTH,RECT_HEIGHT*5-(j+0.5)*RECT_HEIGHT, full_string,fontsize="x-small", color="black", 
                    horizontalalignment='center', verticalalignment='center')
            else:
                ax.text((i+0.5)*RECT_WIDTH,RECT_HEIGHT*5-(j+0.5)*RECT_HEIGHT, pixel_status,fontsize="x-small", color="black", 
                    horizontalalignment='center', verticalalignment='center')
                color = '#666666'
                color_alpha = 1

            ax.text(i*RECT_WIDTH+0.02,RECT_HEIGHT*5-j*RECT_HEIGHT-0.10, pixel_name,fontsize="x-small", color="black")
            ax.add_patch(Rectangle((i*RECT_WIDTH,RECT_HEIGHT*4-j*RECT_HEIGHT), RECT_WIDTH, RECT_HEIGHT, facecolor=color, alpha=color_alpha, edgecolor='black'))
            count+=1
    ax.set_ylim(0,RECT_HEIGHT*5)
    ax.set_xlim(0,RECT_WIDTH*4)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    #plt.axis('off')
    plt.title(run_id + " Visual Report")
    plt.savefig(local_file_save_path+'/'+run_id+'_visual.png')             
    return None


def get_offline_spreadsheet_as_df(spreadsheet_file, range_name, range_cols):
    return pd.read_excel(spreadsheet_file, sheet_name=range_name, usecols=range_cols)

#This is to edit excel with the new data in rows_to_edit
def convert_df_to_xlsx(spreadsheet_file, range_name, range_cols, dataframe, rows_to_edit, add_rows):
    workbook = load_workbook(spreadsheet_file)
    active_sheet = workbook[range_name]
    
    #Should probably do something more robust
    for i in rows_to_edit:
        if add_rows:
            active_sheet.insert_rows(idx=i+2)
        row = dataframe.iloc[i].values
        for j in range (0, len(row)):
            cell = ascii_uppercase[j] + str(i+2) 
            active_sheet[cell]=row[j]
    workbook.save(filename=spreadsheet_file)

#These methods are deprecated for online google sheets accesss
def get_spreadsheet_results(service, spreadsheet_id, range_name):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,range=range_name).execute()
    return result.get('values', [])

def update_spreadsheet(service, spreadsheet_id, range_name, body):
    value_input_option='USER_ENTERED'
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption=value_input_option, body=body).execute()

def getService(pathName, credentialsName, scopes, a1, a2):
    creds = None
    if os.path.exists(pathName):
        with open(pathName, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentialsName, SCOPES_DRIVE)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(pathName, 'wb') as token:
            pickle.dump(creds, token)
    return build(a1, a2, credentials=creds)

if __name__ == '__main__':
        main()
