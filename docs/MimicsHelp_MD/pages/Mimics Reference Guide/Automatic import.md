### Automatic import

To start the Import wizard, first select File and then choose New ProjectWizard. In the File Browser window, you can select where the images to be imported can be found (STEP 1).

Browse to the MedData folder and select the folder called �DICOM_Mandible� in the File browser. The list of files will be displayed in the Filename column and all the files will be automatically selected. Click on one of the files and press CTRL+A to select all files in that folder. Click the Next button.

![](Mimics Reference Guide/../Resources/Images/New_Project_wizard_707x416.png)

An Import log window will show details of the import, including the recognized formats of the file (see the general help files for a list of known formats). 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000278_554x343.png)

Note: In this case the file type is recognized, but in some cases the message log tells you that one or more images are of an unknown file type. If this happens, you have to perform a manual import (see **Import Raw Images**).

Click Next to proceed to the Studies page. This is the second step of the New Project Wizard, in which you need to select the studies to be converted. The study you have just imported is already selected by default.

In this window some information about the project can be found, such as the number of images, pixel size, patient name, orientation parameters, etc. You can also compress your studies to cut off unwanted regions like Air. For this case, we will chose Lossless Compression. 

![](Mimics Reference Guide/../Resources/Images/New_Project_wizard2_431x366.png)

ClickOpenand you will see a progress bar. After the images are successfully imported, you will see a Check Orientation window where you can check and change the orientations of the imported study. Here, the orientation strings L and R stand for Left and Right, A and P stand for Anterior and Posterior, and T and B stand for Top and Bottom respectively. To change the orientation, click on one of the letters and chose the correct orientation from the list. Note that all the other orientation strings are updated automatically.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300027A_346x264.png)

If some orientation is not defined in the DICOMs, you will see an X mark, indicating a missing orientation. You can click on the X mark and assign an orientation to it.

If the orientation of the images is correct, click OK and your Mimics project will open. Now you can process your images using the tools explained in the tutorials **Simon** and **Smart Expand** (Threshold, Region Growing, Edit, etc..)
