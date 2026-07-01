### Creating a 3D representation  
  
In the mask tab you see all created masks listed with their respective threshold. The names of these masks are Green and Yellow. Selecting one mask will make it active.

Now, you still know that the Yellow mask contains the skull, but after a month, when you reload a project, it might be difficult to know in which mask your end result was stored. Therefore, it is advisable to rename the mask (in Project Management, Masks tab). Click on the name Yellow so that it becomes editable; replace Yellow with a more telling name like �skull�.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000287.png)

Click on the Calculate Part button ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/02000288_25x25.jpg).

The Calculate 3D Models dialog box is displayed. Here you can select from which masks you want to calculate the 3D model. To select multiple masks hold the Ctrl key while selecting the other masks. In this case select �skull� and press the Calculate button to generate a Part.

You can set the visualization quality of your model. This is only the visualization on the screen; this parameter does not have any impact on the model that you will actually build on a RP machine!!! Of course, the lower the quality, the less time the program needs to calculate the 3D image and the less memory is needed to load the 3D image afterwards.
