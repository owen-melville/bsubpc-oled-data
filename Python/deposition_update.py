from os import path
import numpy as np
from datetime import datetime
import time
import csv
from io import StringIO
import depo_processor as dp
import composition_analyzer as comp
import pandas as pd
import pathlib
from openpyxl import load_workbook
from string import ascii_uppercase


""" This program takes the raw data from a deposition and indexes it with deposition logs as run online.
Inputs: Master Deposition Spreadsheet
Programs Used: depo_processor.py, compsition_analyzer.py
Outputs: Master Deposition Spreadsheet, Local Data, Local Graphs
"""

UPDATE_MASS_LOGS = True  #Can I add this as command line input?

INPUT_DATA_SHEET = 'Deposition Input'
INPUT_DATA_COLUMNS = 'A:M'

OUTPUT_DATA_SHEET = 'Deposition Output'
OUTPUT_DATA_COLUMNS = 'A:M'

MASS_DATA_SHEET = 'Mass Utilization'
MASS_DATA_COLUMNS = 'A:G'

MATERIAL_DATA_SHEET = 'Evaporator Materials'
MATERIAL_DATA_COLUMNS = 'A:J'

MASTER_DEPOSITION_FILE = 'C:/Users/AM/Desktop/Offline Spreadsheets/Master Deposition List.xlsx'



def main():

    print("Deposition Update Running")
    print("Created by Owen Melville, 2023")
    print("See Python Instructions File for Details")

    input_data_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, INPUT_DATA_SHEET, INPUT_DATA_COLUMNS)
    output_data_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, OUTPUT_DATA_SHEET, OUTPUT_DATA_COLUMNS)
    mass_util_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, MASS_DATA_SHEET, MASS_DATA_COLUMNS)
    material_df = get_offline_spreadsheet_as_df(MASTER_DEPOSITION_FILE, MATERIAL_DATA_SHEET, MATERIAL_DATA_COLUMNS)

    cells_filled_final = len(input_data_df['Deposition ID'].values)
    print("Input Cells filled:", cells_filled_final)

    #Check what rows are in input_data that are not in output_data
    new_row_count = 0
    NUM_REFERENCE_ROWS = 3
    for input_index, input_row in input_data_df.iterrows():
        unique = True
        for output_index, output_row in output_data_df.iterrows():
            if (input_row.iloc[0:NUM_REFERENCE_ROWS].equals(output_row.iloc[0:NUM_REFERENCE_ROWS])):
                unique = False #We've found a match!
                break
        if (unique):
            print ("Found new deposition: " + str(input_row.iloc[0]) + ", " + str(input_row.iloc[1]))
            new_row = {'Deposition ID': input_row.loc['Deposition ID'], 'Material':input_row.loc['Material'],
             'Deposition Type':input_row.loc['Deposition Type'], 'Deposition Subtype':input_row.loc['Deposition Subtype'], 
             'Processing Status':'Unprocessed'}
            output_data_df.loc[len(output_data_df)] = new_row #Add the new row
            output_data_df = output_data_df.where(pd.notnull(output_data_df), "TBD") #Replace NaN values with TBD
            new_row_count += 1

    if new_row_count > 0: #Currently this convert_df_to_xlsx function does not work... replaces whole file
        convert_df_to_xlsx(MASTER_DEPOSITION_FILE, OUTPUT_DATA_SHEET, OUTPUT_DATA_COLUMNS, output_data_df, range(cells_filled_final-new_row_count, cells_filled_final), True) 
        print("Cells added:", new_row_count)
    else:
        print("No new input rows")

    rows_to_edit = []
    num_material_file = 1
    updated_cells = 0
    mass_utilization_update_rows = []
    update = False
    for output_index, output_row in output_data_df.iterrows(): #Check our deposition data for unprocessed runs
        try:
            update = output_row.loc["Processing Status"] != 'Processed' #Check if we need to process the data
        except:
            update = False

        if (num_material_file>1): #This weird setup is to stop the program from running 3x for the same depo with 3 materials
            update = False
            num_material_file-=1

        if (update): #We are given the go-ahead to update
            deposition_id = output_row.loc["Deposition ID"]
            rows_to_edit.append(output_index)

            input_rows = input_data_df.loc[input_data_df['Deposition ID'] == deposition_id]
            num_material_file = len(input_rows)
            print("\nProcessing Run: " + deposition_id)
            print("Number of Materials: ", num_material_file)

            file_name = input_rows.loc[: , 'Log File Name'].values[0]
            raw_data_file_path = str(pathlib.Path().resolve()).replace("Python", "IC6 Evaporation Logs") + "/" + file_name

            if path.exists(raw_data_file_path):
                print("File found: " + file_name)

                mat_names = input_rows.loc[:, 'Material'].values #Get the names of the materials
                mat_sensors = input_rows.loc[:, 'Sensor'].values #Get the sensors of the materials

                #Create a local folder to save files and graphs
                local_file_save_path = str(pathlib.Path().resolve()).replace("Python", "Output Data") + "/" + deposition_id
                if path.exists(local_file_save_path):
                    print("Local directory already exists...")
                else:
                    print("Creating Local Directory: " + deposition_id)
                    os.mkdir(local_file_save_path)

                #This function condenses the raw data, analyzes it, and creates a few graphs
                t_gross, t_dep, condensed_data, avg_rates, rate_stdevs, avg_powers, times_spent_tot, times_spent_sub = process_file(
                    deposition_id, raw_data_file_path, local_file_save_path, mat_names, mat_sensors, num_material_file)

                condensed_df = pd.DataFrame(condensed_data[1:], columns=condensed_data[0])
                condensed_df.to_csv(local_file_save_path+"/"+deposition_id+"_condensed.csv", sep='\t', encoding='utf-8')

                #OK its time to try to update the output_data spreadsheet
                for mat in range (0,num_material_file):
                    error_occured = False
                    today_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    try: #If we got None back from process_file, this will fail and we will get error_occured = True
                        t_gross_str = t_gross[mat]*1000
                        t_dep_str = t_dep[mat]*1000
                        avg_rate = avg_rates[mat]
                        rate_stdev = rate_stdevs[mat]
                        avg_power = avg_powers[mat]
                        time_spent_tot = times_spent_tot[mat]
                        time_spent_sub = times_spent_sub[mat]
                    except:
                        error_occured = True
                    if error_occured == False: 
                        update_values = ['Processed', today_string, t_gross_str, t_dep_str, avg_rate, rate_stdev, avg_power, time_spent_tot, time_spent_sub]
                    else:
                        update_values = ['File Parse Error', today_string, "TBD", 'TBD', "TBD", "TBD", "TBD", "TBD", "TBD"]

                    for i in range (0, len(update_values)): #Update the dataframe
                        output_data_df.iloc[output_index+mat, NUM_REFERENCE_ROWS+i+1]=update_values[i] #Plus 1 is due to the ref_rows change

                    #This is the entry generated for each material in the file for the mass log <--- Update later
                    today_string = "Autogenerated on: "+ datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                    generated_row=[today_string]
                    mass_utilization_update_rows.append(generated_row)

                #Update the mass utilization logs <--- Update later
                if UPDATE_MASS_LOGS and error_occured == False:
                    print("Updating mass logs...")
                    for mat in range (0, num_material_file):
                        try:
                            material_used = t_gross[mat]*1000
                            material_usage_rate = material_df.loc[material_df['Material']==mat_names[mat],'Usage Rate (A/mg)'].values[0]
                            mass_used = float(material_used) / float(material_usage_rate) / 1000 #Amount in g
                            last_mass = mass_util_df.loc[mass_util_df['Material']==mat_names[mat],'Final Mass (g)'].values[-1]
                            last_thickness = mass_util_df.loc[mass_util_df['Material']==mat_names[mat],'Final Remaining Thickness (A)'].values[-1]
                            new_mass = float(last_mass) - float(mass_used)
                            new_thickness = float(last_thickness) - float(material_used)
                            
                            mass_util_df.loc[len(mass_util_df)]=[datetime.now().strftime("%Y/%m/%d"),
                                mat_names[mat], "Deposited", last_mass, new_mass, last_thickness, new_thickness]

                            convert_df_to_xlsx(MASTER_DEPOSITION_FILE, MASS_DATA_SHEET, MASS_DATA_COLUMNS, mass_util_df, [len(mass_util_df)-1], True) 
                        except Exception as e:
                            print("Can't update mass log for material: ", mat_names[mat])
                            print("Error: ", e)
                            print("If the above error says cannot convert string Unknown, then the mass utilization data is not in the reference spreadsheet")
  
            else: #We can't find the file sadly
                for i in range (0, num_material_file):
                    output_data_df.loc[output_index+i,"Processing Status"] = "File Not Found"
                print("File not found: " + file_name)
    
    #Update the spreadsheet again
    if update:
        convert_df_to_xlsx(MASTER_DEPOSITION_FILE, OUTPUT_DATA_SHEET, OUTPUT_DATA_COLUMNS, output_data_df, rows_to_edit, False) 
        print("Update Complete")
    else:
        print("No Update Required")

def process_file(deposition_id, raw_data_file_path, local_file_save_path, mat_names, mat_sensors, num_materials):
    condensed_data = dp.get_condensed_data(raw_data_file_path, mat_names, mat_sensors) #get condensed data
    processing_results,zero_counts = dp.get_material_thicknesses_and_plotnames(mat_names, condensed_data, deposition_id, local_file_save_path) #get thicknesses and plotnames

    #Just reformatting the data to a different structure
    thickness_gross = []
    thickness_deposited = []
    plot_names = []
    avg_rates = []
    rate_stdevs = []
    avg_powers = []
    times_spent_tot = []
    times_spent_sub = []
    if processing_results != None:
        for material in processing_results: #These aren't magic numbers, just organizing ones
            thickness_gross.append(material[0])
            thickness_deposited.append(material[1])
            plot_names.append(material[2])
            avg_rates.append(material[3])
            rate_stdevs.append(material[4])
            avg_powers.append(material[5])
            times_spent_tot.append(material[6])
            times_spent_sub.append(material[7])
    else:
        print("Error getting thickness/power/rate columns from condensed data")
        return None, None, None, condensed_data, None
    print("Thicknesses:",thickness_gross,"Deposited:", thickness_deposited)
    print ("Rates:", avg_rates, "Stdevs:",rate_stdevs)
    print ("Average Powers: ", avg_powers)

    comp_file_name=""
    #This analysis needs to return the composition too! Currently just returns the graphs
    if num_materials > 1 and len(zero_counts) > 1:
        comp_file_name = comp.get_composition_graphs(deposition_id, local_file_save_path,
            condensed_data, mat_names, thickness_deposited, zero_counts)
        print(comp_file_name + " Saved Locally")
    elif len(zero_counts)<2:
        print("Issue getting zeroes for composition determination")
    elif num_materials == 1:
        comp_file_name = None

    return thickness_gross, thickness_deposited, condensed_data, avg_rates, rate_stdevs, avg_powers, times_spent_tot, times_spent_sub

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


#These methods are deprecated, for online communication with google sheets
def get_spreadsheet_results(service, spreadsheet_id, range_name):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,range=range_name).execute()
    return result.get('values', [])

def update_spreadsheet(service, spreadsheet_id, range_name, body):
    value_input_option='USER_ENTERED'
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_name,
        valueInputOption=value_input_option, body=body).execute()

def get_service(token_path, cred_path, version, type, scopes):
    creds = None
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                cred_path, scopes)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    service = build(type, version, credentials=creds)
    return service

if __name__ == '__main__':
    main()
