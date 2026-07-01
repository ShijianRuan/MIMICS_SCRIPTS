### Split Mask

The Split Mask tool is part of the **Segment** menu and allows splitting of a single mask into two separate masks. To open the tool go to _Segment > Split Mask_ in the menu or click on the ![](Mimics Reference Guide/../Resources/Images/Split%20Mask%20icon_25x41.png) icon in the Segment toolbar. This tool allows easy and quick separation of anatomical parts e.g. heart from the surrounding rib cage or separating talus and calcaneus in the foot.

![](Mimics Reference Guide/../Resources/Images/SplitMask_Workflow-01_800x386.png)

The Split tool requires two marked regions to be used as input for splitting the selected mask. Region A is active by default and marking is possible on the axial, coronal and sagittal viewport. Start by marking Region A and then click the Region B button to mark the second region.

![](Mimics Reference Guide/../Resources/Images/Region_A%20Add_55x59.png) |  ![](Mimics Reference Guide/../Resources/Images/Region_B%20Add_54x57.png) |  ![](Mimics Reference Guide/../Resources/Images/Region_A&B Remove_54x57.png)  
---|---|---  
Mark region A | Mark region B | Unmark (press Alt key)  
  
Click on OK to start the splitting operation. After the operation is complete the dialog box will automatically close and the result will be added to the Masks tab as Region A and Region B.

In most of the cases the Split Mask tool gives good results with markings on just single slice. It is not mandatory to mark on multiple slices, but this can improve the result in case of undesired splitting.

_Diameter_

To resize the diameter of the brush, use the diameter slider or hold the CTRL button and left mouse button down while moving the mouse to the left (decrease) or right (increase).

**Unmark**

To unmark the region click on the Alt key and a minus sign will appear at the top right of the brush circle.

**Advanced options**

The Split Mask tool allows using existing masks as input for Region A and Region B. Click on the icon at the right of the entity selection box to expand the list of existing masks in the project and select one of the existing masks.

![](Mimics Reference Guide/../Resources/Images/SplitMask_AdvancedOptions.png)

The Split Mask tool also allows saving the marked regions as separate masks. Tick ON the checkbox of �Save masks of input regions� in the advanced options to save the input as separate masks. After the operation is complete the input masks will be saved in the Masks tab with suffix _input along with results of the Split Mask operation.
