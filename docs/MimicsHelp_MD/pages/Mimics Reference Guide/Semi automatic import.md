### Semi-automatic import

Now we will try to import the Bitmap images, you can find the dataset in the folder "BMP_Leg" in your MedData directory. Select File > New Project Wizard and browse to the C:\MedData\DemoFiles\BMP_Leg directory. Click on one of the images in BMP_Legfolder and press Ctrl+A on your keyboard to select all files in it. Press the Next button and the Import Log will be displayed. Click Next to see the Images Properties dialog. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300027C_554x343.png)

Here you can preview your images and order them according to your preferences. You can also check if the scan resolution is correctly read. Uncheck force isotropic sampling checkbox and change the Z direction to 1. You can also change the dimensions of your images. This information will be typically provided by the radiologist who took the scan. Correct values should be entered here to ensure correct dimensions of the volumes and the Parts that will be created further on. Leave it in mm scale for this case and click Next.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0300027D_554x343.png)

In the Edit Images dialog, you have the option to crop the images or resample them. For this example, we will leave the values as is and click Next.

Now you should be able to set the orientation parameters as described in the previous paragraph and calculate a good 3D.
