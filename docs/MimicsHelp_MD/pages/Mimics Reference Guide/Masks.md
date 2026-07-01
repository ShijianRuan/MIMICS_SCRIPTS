#### Masks  
  
![](Mimics Reference Guide/../Resources/Images/PMTab_Mask_486x227.png)

A mask is a collection of pixels where all actions (editing, region growing, ..) and calculations (3D calculations, STL, ..) are based on.

##### List of the created masks

Name |  Name of the mask. By clicking on the name of the mask, it can be renamed.  
---|---  
Visible |  Lists if the mask is visible or not by means of glasses.  
Lower Threshold |  Lower Threshold setting of the mask.  
Higher Threshold |  Higher Threshold setting of the mask.  
_Assembly_ | Add the mask to a non manifold assembly. The first mask that is added to the assembly will remain unchangeable. When a mask is added to the assembly and intersects with the first mask, the intersecting regions will be substituted from the last mask added to the assembly. To undo this action and restore the mask, you need to click **Edit** and **Undo**. For further information about the non manifold assembly, please refer to this [section](<Calculate%20Non%20Manifold.md>) of the help menu.  
Images | The link to the image set. The link of a mask to an image set cannot be modified  
  
##### Functions on masks

New |  Creates a new mask. On create new mask the threshold bar will pop up, allowing you to immediately set a threshold or select one of the predefined thresholds.  
---|---  
Delete |  Deletes the selected mask  
Properties |  Gives numerical information of the selected mask  
Duplicate |  Duplicates the selected mask  
Clear |  Clears the contents of the selected mask, the threshold of the mask is kept  
Calculate Part |  Opens the Calculate Part window  
Action |  Lists the available function on the selected mask  
  
##### Properties

The mask properties give numerical information about the gray values in the selected mask: 

![](Mimics Reference Guide/../Resources/Images/Mask_Properties.png)

Threshold low |  Lower threshold value used for the Parts calculation  
---|---  
Threshold high |  Higher threshold value used for the Parts calculation  
Minimum value |  Minimum gray value in the selected mask  
Maximum value |  Maximum gray value in the selected mask  
Average value |  Average gray value from the selected mask   
Standard deviation |  ![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/03000059.png)  
Number of pixels |  Amount of pixels in the selected mask   
Mask volume |  The volume of the mask  
Scale |  The scale you are working in: GV (Grayvalues) or HU (Hounsfield Units).  
  
_Note_ : The number of masks that could be created up till Mimics 17.0 was limited to 32. From Mimics 18.0 onwards there is no limitation on the number of masks.
