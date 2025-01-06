import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pathlib
from openpyxl import load_workbook
from string import ascii_uppercase

OLED_OUTPUT_SHEET = 'OLED Output'
OLED_OUTPUT_COLUMNS = 'A:P'

OLED_COMPARE_INPUT_SHEET = 'OLED Comparison'
OLED_COMPARE_INPUT_COLUMNS = 'A:D'

MASTER_OLED_FILE = 'C:/Users/AM/Desktop/Offline Spreadsheets/Master OLED Log.xlsx'

#These are the categories we want to make graphs from
RESULT_CATEGORIES = ['Voltage(V)',  'Current(A)',  'J(mA/cm2)',   'PE(lm/W)', 'EQE(%)',  'L(cd/m2)', 'CE(cd/A)'] #Note, added category
#How we want to plot these categories
RESULT_SCALE = ['linear', 'log', 'log', 'linear', 'linear', 'log', 'linear']
#Minimum values that we want to plot for these values
MIN_VALUE = [0, 1E-5, 1, 0, 0,1,0]
DESIRED_PAIRINGS = [[0,2,False],[2,5,False], [2,4,True], [2,3,True],[2,6,True],[0,5,False],[5,3,True], [5,4,True],[5,6,True]]

def main():
    print("OLED Comparison Running...")
    print("Created by Owen Melville, 2023")
    print("See Python Instructions File for Details")

    #Get Comparison Requests
    compare_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_COMPARE_INPUT_SHEET, OLED_COMPARE_INPUT_COLUMNS)

    #Get OLED Output Data
    oled_output_df = get_offline_spreadsheet_as_df(MASTER_OLED_FILE, OLED_OUTPUT_SHEET, OLED_OUTPUT_COLUMNS)

    #Check for each required comparison
    rows_to_edit = []
    for compare_index, compare_row in compare_df.iterrows():
        if compare_row['Status']!='Complete':
            rows_to_edit.append(compare_index)
            id_list = compare_row['ID List'].split(',')
            print(id_list)
            id_string = compare_row['ID List'].replace(",", "_")
            legend_basic = compare_row['Legend'].split(',')
            comparison = compare_row['Comparison']

            print ("Comparing devices: ", id_string)

            spectral_xs = []
            spectral_ys = []
            spectral_ystdevs = []
            condensed_data = []
            median_colors = []
            legend = []
            v_ons = []
            data_obtained = False
            try:
                for i in range (0, len(id_list)):#Get the raw data
                    device_id = id_list[i] 
                    output_data_path = str(pathlib.Path().resolve()).replace("Python", "Output Data") + "/" + device_id + '/Raw Data'
                    spectral_df = pd.read_csv(output_data_path+'/'+device_id+"_avgspectra_250cdm2.csv", delimiter='\t') 

                    legend.append(device_id+": "+ legend_basic[i])
                    legend.append("Standard Deviation")

                    spectral_xs.append ( spectral_df['Wavelength (nm)'].values )
                    spectral_ys.append ( spectral_df['Intensity (Counts)'].values )
                    spectral_ystdevs.append ( spectral_df['Standard Deviation (Counts)'].values )
                    
                    condensed_data .append ( pd.read_csv(output_data_path+'/'+device_id+"_avgoutput.csv", delimiter='\t') )
                    median_colors.append(oled_output_df.loc[oled_output_df['OLED ID'] == device_id]['Median Color at 250 cd/m2'].values[0])
                    v_ons.append(oled_output_df.loc[oled_output_df['OLED ID'] == device_id]['Median Turn-On Voltage (V)'].values[0])

                data_obtained = True
            except Exception as e:
                print("Error getting data files: ", e)

            BASIC_COLORS = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']
            rgb_colors = []
            for color in median_colors:
                h = color.lstrip('#')
                rgb_colors.append (tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) )
            if colors_are_close(rgb_colors,0.3): #This threshold is uncertain... more of an optical phenomena
                median_colors = BASIC_COLORS[0:len(median_colors)]

            if data_obtained:
                #Create Comparison Output Folder if it doesn't exist
                local_file_save_path = str(pathlib.Path().resolve()).replace("Python", "Output Data") + "/Comparisons/" + id_string
                if os.path.exists(local_file_save_path):
                    print("Local output directory already exists...")
                else:
                    print("Creating Local Output Directory: " +id_string)
                    os.mkdir(local_file_save_path)

                print("Creating Spectral Graphs")
                #Spectral graphs
                create_smeared_graph(spectral_xs, spectral_ys, spectral_ystdevs, local_file_save_path, legend, 
                    'Wavelength (nm)', 'Intensity (counts)', comparison + ": Emission Spectra", 'Compared Emission Spectra',
                         'linear', 'linear', median_colors, 380, 0)    

                print("Creating Optoelectronic Graphs...")
                #Other graphs 
                for pairing in DESIRED_PAIRINGS:
                    pairing_xindex = pairing[0]
                    pairing_yindex = pairing[1]
                    use_cutoff = pairing[2]     

                    xs = []
                    ys = []
                    y_vars = []

                    x_scale = RESULT_SCALE[pairing_xindex]
                    y_scale = RESULT_SCALE[pairing_yindex]
                    x_min = MIN_VALUE[pairing_xindex]
                    y_min = MIN_VALUE[pairing_yindex]
                    x_label = RESULT_CATEGORIES[pairing_xindex]
                    y_label = RESULT_CATEGORIES[pairing_yindex]

                    title = comparison + ": " + x_label.split('(')[0]+ ' vs ' + y_label.split('(')[0]
                    file_label = "Comparison_" + x_label.split('(')[0]+ '_' + y_label.split('(')[0]


                    for i in range (0, len(condensed_data)):
                        df = condensed_data[i]
                        v_on = float(v_ons[i])
                        voltage_list = df['Voltage(V)'].values
                        stdev_list = df[y_label + " stdev"].values
                        von_index = next(x[0] for x in enumerate(voltage_list) if x[1] > v_on)
                        min_index = 0 #This algorithm is not foolproof
                        if use_cutoff:
                            max_value = np.amax(df[y_label].values[von_index:])
                            for i in range (0, len(stdev_list)):
                                if max_value-stdev_list[i] > 0 and max_value-stdev_list[i+1] > 0:
                                    min_index = i
                                    break
                    
                        x = df[x_label].values[min_index:]
                        y = df[y_label].values[min_index:]

                        y_var = stdev_list[min_index:]
                        xs.append(x)
                        ys.append(y)
                        y_vars.append(y_var)

                    create_smeared_graph(xs,ys,y_vars,local_file_save_path,legend,x_label,y_label,title,file_label,x_scale,y_scale,median_colors,x_min,y_min)    

            compare_row['Status']='Complete'

    #Update the spreadsheet again
    #upload = [compare_df.columns.values.tolist()] + compare_df.values.tolist()
    #body = {'values': upload}
    #update_spreadsheet(sheets_service, OLED_FILE_ID, OLED_COMP_RANGE, body)
    convert_df_to_xlsx(MASTER_OLED_FILE, OLED_COMPARE_INPUT_SHEET, OLED_COMPARE_INPUT_COLUMNS, compare_df, rows_to_edit, False)
    print("Update Complete")

def colors_are_close(rgb_colors, closeness_threshold):
    pairs = [(a, b) for idx, a in enumerate(rgb_colors) for b in rgb_colors[idx + 1:]]
    for pair in pairs:
        item_1 = np.array(pair[0])/255
        item_2 = np.array(pair[1])/255

        distance = np.linalg.norm(item_1-item_2)
        
        if distance < closeness_threshold:
            return True
    return False
                
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

def create_smeared_graph(xs, ys, y_vars, local_file_save_path, legend, x_label, y_label, title, file_label, x_scale,y_scale, median_colors, x_min, y_min):
    fig, ax = plt.subplots() 
    for i in range (0, len(xs)):
        x = xs[i]
        y = ys[i]
        median_color = median_colors[i]
        y_var = y_vars[i]
        ax.plot(x, y, color=median_color)
        ax.fill_between(x, y+y_var,y-y_var,alpha=0.3, color=median_color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.yscale(y_scale)
    plt.xscale(x_scale)    
    plt.xlim(xmin=x_min)
    plt.ylim(ymin=y_min)
    plt.legend(legend)
    plt.savefig(local_file_save_path+'/'+file_label+'.png')
    plt.clf()
    plt.close()

if __name__ == '__main__':
        main()
