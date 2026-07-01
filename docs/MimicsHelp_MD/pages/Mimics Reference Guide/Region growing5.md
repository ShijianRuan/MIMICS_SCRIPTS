### Region growing

Now that the box is delimited by the layers removed from the active mask, a region growing can be performed to get the obturator into a new mask. Go to an axial image that has a position between 362,00 and 387,00. This is to make sure that the starting pixel for the region growing lies within the region of interest. Press the Region grow button and click in the axial image within the box. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002CE_300x232.png)

Boundaries of the mouth cavity in the axial image.

The mouth cavity is now within a new mask. In the figure below you clearly see the mouth cavity within the active (blue) mask from the axial and the sagittal viewpoint. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002CF_300x232.png) ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002D0_364x232.png)

Axial and sagittal view of the mouth cavity

Because we disconnected the pixels of the mouth cavity from the other pixels in the original mask, the region growing was confined to the region of interest. 

Press the Calculate Part button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002D1_25x25.jpg) and select the mask of the mouth cavity. Choose custom quality and press the Calculate button. The processing of the 3D model is started. 

On the right of the Part you see a toolbar and a button where you can select some predefined viewpoints for your 3D model ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002D2_25x25.jpg). If you press the bottom view you should obtain a model as shown in the figure below. You can also enable transparency using the ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002D3_25x25.jpg) button.
