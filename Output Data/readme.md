<h1>Output Data Readme</h1>

This folder contains all the results of the automated analysis in the <b>Python</b> folder. There are two sorts of folders: Deposition output folders and OLED output folders. Deposition output folders contain the results of analysis of the deposition runs (from the IC6) and end in the suffix <b>-L#</b> where # ranges between 1 and 6 representing the different layers in a single OLED. OLED output folders contain the analytical results of the optoelectronic data in the <b>Raw OLED Data</b> folder and do not contain this suffix (eg OLED-033). 

<h2>1. Deposition Output Folders</h2>

<b>Deposition Graphs:</b>

These graphs show the input power, deposition rate, and total thickness as a function of time. The shutter was open only between two zeroes in the thickness data, which can be seen below as between 500s and 700s, approximately. This is how the thickness of the layer is calculated. 

![OLED-004-L3_TxOPhCz](https://github.com/user-attachments/assets/703a69bf-c994-4cfb-92c8-7cb5d460867e)

<b> Composition Graphs: </b>

These graphs show the distribution of composition (by rate) within different thickness bins in each emissive layer. You will only see these in devices with a Host and one or more Dopant (typically <b>-L3</b>)

![OLED-015-L3 Composition](https://github.com/user-attachments/assets/fe50ee8c-1577-408d-b8ec-9f0ad1c72b6e)

<b> Condensed Data: </b> 

This takes the IC6 readings, which contains excessive information, and parses it into a smaller set of more useful data. 

<h2>2. OLED Output Folders</h2>

<b>Visual Chart:</b>

This chart summarizes the performance of each pixel. Pixels that were short circuited or open circuit or were untestable (often A5 due to the testing setup) are greyed out. The color displayed is an approximation of the color of the device by converting the spectra into RGB, but it is not a true representation of the color. 

![OLED-033_visual](https://github.com/user-attachments/assets/a7bf1c51-0673-419e-9e33-9deae7a9bbaf)

<b>Average Spectra:</b>

This graph shows the shape of the emission at a specified luminance, averaged across pixels. The blurring of the line represents pixel-to-pixel variation.

![OLED-033_average_spectra](https://github.com/user-attachments/assets/f29e4439-3bbf-4b59-9094-68e31de2c22d)

<b> Energy Diagram: </b>

This diagram shows the energy levels of the materials in the device.

![OLED-033 Energy Level Diagram](https://github.com/user-attachments/assets/302d4781-5f28-46b0-8755-c6ed0e65447b)

<b> CIE Plot: </b>

This plot shows the CIE coordinates of the pixels. The color rendered is not exactly accurate. 

![OLED-033 CIE Plot](https://github.com/user-attachments/assets/1e5aab4c-d9b1-4363-b691-7a9af5b6288d)

<b> Raw Data Folder </b>:

This folder actually contains condensed or calculated output data that was saved for convenience. 

<b> Parameter Graphs Folder </b>:

This folder contains many graphs that correlate one parameter (eg Voltage) against another (eg Luminant Intensity). Sometimes the data shows dots, which is the set of points, other times it shows a blurred line, which is to simplify the depiction of the data while still showing the variation.

![OLED-033_Voltage_vs_L](https://github.com/user-attachments/assets/1f473366-0ad0-45b1-bbb4-e9f919665814)

<b> Deconvolution Folder/ Sample Deconvoluted Spectra </b>

This spectra shows how well the program was able to break apart the composite emission into its 2 components visually.

![Sample Deconvolution Spectra](https://github.com/user-attachments/assets/f3352836-d6e1-43b4-a1a3-2030606b86b8)

<b> Deconvolution Folder/ Deconvolution Error </b>

This graph shows the R^2 accuracy of the spectral deconvolution as a function of voltage.

![Deconvolution Error](https://github.com/user-attachments/assets/82b2162e-91a4-497c-bec3-24bb101aacdf)

<b> Deconvolution Folder/ Deconvolution Graph </b>

This graph shows how the two contributions (eg flurorescent dopant and TADF assistant) vary as a function of voltage. 

![OLED-033_deconvolution](https://github.com/user-attachments/assets/1380d997-350a-418d-b618-cb2360d2f910)

<b> Deconvolution Folder/ Hyperfluorescence Graph </b>

This graph compares the set of OLEDs with FD-only, TD-only, and TD+FD to visualize the role of the TD in boosting the emission of the FD (Hyperfluorescence).

![Hyperfluorescence_Spectral_Comparison](https://github.com/user-attachments/assets/2dcf9b39-d530-4f59-b909-467fcfa70131)

<b> Deconvolution Folder / FD Tradeoff </b>

Often, color purity and current efficiency come at each other's expense within a set of OLED pixels. This graph attempts to visualize that. 

![FD_tradeoff](https://github.com/user-attachments/assets/8d31d52c-71c0-4e07-8b6c-e0b640898b92)

<b> Correlations Folder </b>

This folder contains all the correlation data between material thicknesses in a set of OLEDs and the performances of the pixels. Some of the top correlations are graphed, by default, regardless of whether a physical connection makes sense. Usually, when one parameter is tightly correlated to a specific material, it is also tightly correlated to other material thicknesses. This is because the sources are in a set geometric pattern, and so the thickness of certain layers may be highly correlated or anticorrelated depending on the location. 

![FD_%_top_correlation_deconv](https://github.com/user-attachments/assets/0f8dfb3d-e6fe-4864-880f-cb8ffa9be079)

