<h1>Python Programs</h1>

The main programs used are "Deposition Update", which was used to analyze the deposition data and "OLED Update", which was used to analyze both the deposition data and the optoelectronic data. The results of these programs are in the <b>Output Data</b> folder.

<h2>Deposition Update</h2>

This program takes the deposition data in the <b>IC6 Evaporation Logs</b> folder and logs the thicknesses of each layer of the OLED into the <b>Master Deposition List</b> in the <b>Offline Spreadsheets</b> folder. It also plots the rate and thickness as a function of time, storing that data in the <b>Output Data</b> folder. The end of a layer is usually triggered by a manual zeroing of the thickness which can be seen in the deposition run data. However, in some cases there was an error with the manual zeroing (or in the case of Aluminum, the tooling), so the the thickness recorded in the spreadsheet was entered manually.

<h2>OLED Update</h2>

This program takes the optoelectronic data from the OLEDs in the <b>Raw OLED Data</b> folder and calculates various performance metrics, generates opto-electronic performance plots, and performs correlative analysis between performance metrics and the thicknesses of the different layers. All of these results are stored in the <b>Output Data </b> folder and the <b>Master OLED Log</b> spreadsheet in the <b>OFfline Spreadsheets</b> folder. The program also determines what data to disregard (eg short circuited or open circuited devices). 
