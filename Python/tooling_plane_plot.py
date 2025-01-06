import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import csv

import os.path
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def main():
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    data = []
    first_row = True
    with open ('OLED_pixel_positions.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in reader:
            if first_row == False:
                data.append(row)
            first_row = False
    data = np.array(data)
    x = np.unique(np.transpose(data)[1]).astype(float)
    y = np.unique(np.transpose(data)[2]).astype(float)
    (xs, ys) = np.meshgrid(np.unique(x), np.unique(y))

    #In the future, this code will all go in the parent program
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
    service = get_service('token.pickle', 'credentials.json', 'v4', 'sheets', SCOPES)

    tooling_data = get_spreadsheet_results(service, '1zbHL38qqhw4z8ziA1ul2V1xXfp6MkvA1gWshWFoa72Y', 'Sheet1!A1:N')
    #In the future, we will get only this tooling_data from the parent program


    #Can use column references instead of fixed numbers
    for tooling_row in tooling_data[1:]:
        A = float(tooling_row[11]) #11
        B = float(tooling_row[12]) #12
        C = float(tooling_row[13]) #13
        zs = A*xs + B*ys + C

        name = tooling_row[2] + " " + tooling_row[3] + " " + tooling_row[5]
        ax.plot_surface(xs, ys, zs, cmap=cm.gray)
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=name)

        for row in data:
                x_i = float(row[1])
                y_i = float(row[2])
                z_i = A*x_i + B*y_i + C
                label = row[0]
                ax.text(x_i,y_i,z_i, label, color='red')

        fig.tight_layout()
        plt.savefig(name+'.png')
        plt.cla()

def get_spreadsheet_results(service, spreadsheet_id, range_name):
    sheet = service.spreadsheets()
    result = sheet.values().get(spreadsheetId=spreadsheet_id,range=range_name).execute()
    return result.get('values', [])

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
