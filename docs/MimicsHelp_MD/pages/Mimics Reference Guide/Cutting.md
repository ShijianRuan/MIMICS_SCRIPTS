### Cutting  
  
After the calculation of the 3D you will see a 3D representation of the Skull mask. To be able to make a cut that fits well, make the skull transparent by clicking on the ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002DF_25x25.jpg) button and choose to view the skull from the Right view. Now you can pan and zoom so you can see the hole clearly.

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002E0_213x179.png) |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030002E1_214x180.png)  
---|---  
  
If you then zoom and pan, you can clearly view the hole in the skull through the intact side. 

![](Mimics Reference Guide/../Resources/Images/skull_with_hole%203d_simulation_460x282.bmp)

This way we can easily draw around this hole. To do this use _Cut with Polyplane_ option, go to **3D Tools > Cut > With Polyplane** in the menu. You will see following dialog:

![](Mimics Reference Guide/../Resources/Images/Cut_with_polyplane_simulation.png)

Select the 3D from the skull in the Objects to Cut list. The New button is already enabled so we can immediately start drawing a cutting path. Do this by clicking several times with your left mouse button around the hole like below. To end the drawing, double click with your left mouse button.

![](Mimics Reference Guide/../Resources/Images/skull_with_hole%203d_simulation2_576x352.bmp)

You can see that a cutting path has been added to the cutting path list. You can now make the 3D opaque again by clicking on the ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/020002DF_25x25.jpg) button. You can then rotate the 3D to determine if the cut went through the whole skull or not:

![](Mimics Reference Guide/../Resources/Images/skull_with_hole%203d_simulation3_578x542.bmp)

As you can see, it would be best if we adjust the depth of the cutting path. You can do this by clicking on the Properties button while the cutting path is selected. This will open the cutting path properties dialog:

![](Mimics Reference Guide/../Resources/Images/Cutting_plane_properties_simulation_332x266.png)

Adjust the Depth of the cutting path from 20.0mm to 30.0mm and enable the Closed checkbox (this will close the cutting path). Click on Preview to view the result. When you are happy with the result, close the Cutting Path Properties by clicking on the OK button. Enable the Keep Originals checkbox (since we want to keep the original 3D) and finish the cut by clicking on the OK button of the Cut with Polyplane tool.

You can see in the 3D objects list that a new 3D object was added.
