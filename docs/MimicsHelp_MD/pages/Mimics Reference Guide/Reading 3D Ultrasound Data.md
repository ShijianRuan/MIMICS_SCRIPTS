# Reading 3D Ultrasound Data

The New Project Wizard provides the functionality of importing 3D Ultrasound images. The steps of importing the DICOMs are similar to the standard DICOM import as mentioned in the previous section. After you have chosen the file(s)/folder to be imported the following import dialog box will appear.

![](Mimics Reference Guide/../Resources/Images/3D_US_Import_617x406.png)

Press **'Convert'**.

If the pixels in the dataset are rectangular you will receive the following message.

![](Mimics Reference Guide/../Resources/Images/Rectangular_Pixels.png)

If you chose �Reslice images�, the anatomical proportions will be preserved, however the dimensions and greyvalues of the dataset will be recalculated and interpolated. It is recommended to use when the difference between the sides is rather big (e.g. 0.5x0.7).

If you chose �Resize images�, initial grey values will be preserved, however the dataset will be visually stretched and measurements will be influences. It can be picked if the deviation between width and height is very small (e.g. 3.9999999 and 4.0).

You can select the desired pixel size using the dropdown box. The action will be applied after you press 'OK'. If the checkbox �Apply to all studies with same pixel sizes� is checked, the settings will be repeated for all studies selected in the Import Wizard.

In case of square pixels, Mimics will directly come to the orientation window. Here you will have to define the orientation of the imported ultrasound images.

![](Mimics Reference Guide/../Resources/Images/3D_US_Import5_orientation_box_599x456.png)

After you have defined the orientation of the images, you will have successfully imported 3D ultrasound images.

![](Mimics Reference Guide/../Resources/Images/3D_US_Import_result_621x347.png)
