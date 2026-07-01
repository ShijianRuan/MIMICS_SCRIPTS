#### Import the Patran volume mesh in Mimics

You can import a Patran neutral file by clicking on the Import Mesh button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020001FF_25x25.jpg) in the FEA project management tab:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000200_236x217.png)

This will open a file browser:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000201_554x312.png)

Change the type of the listed files to Patran Neutral and all Patran files will be shown. Then browse to the directory where the files are located, select the correct file and click on the Open button, the files will be imported.

##### Supported element types

The Mimics FEA module supports four types of Patran elements:

  * 5 (4-node tetrahedron)


  * 7 (6-node wedge)


  * 8 (8-node hexahedron) 


  * 5 (10-node quadratic tetrahedron)


Remark: If you require other element types, please let us know and we will try to implement those elements in our future releases.

##### Supported Patran packets

The Mimics FEA module supports 6 Patran packets:

  * 25 (File title)


  * 26 (Summary data)


  * 1 (Node data)


  * 2 (Element data)


  * 3 (Material properties, export only)


  * 4 (Element properties, export only)


Material properties: Patran writes out 96 material properties. Mimics assigns only 7 of them, the others are set to 0. Exported material properties: 2 (density), 27-29 (E-modulus), 30-32 (Poisson coefficient). For E-modulus and Poisson coefficient all three properties contain the same value.
