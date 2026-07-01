### Dynamic Region Growing

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/DynamicRegionGrowingPulmo.png)

The **Dynamic Region Growing** tool allows you to segment an object based on the connectivity of the gray values in a dynamically selected gray value range.

The creation of this new mask starts when you select a pixel. Mimics starts comparing the gray values of the neighboring pixels. The pixels with gray values that obey the following rule will be added to the new mask. 

|� -i | < d  with 

� the average gray value 

i the new gray value 

d the deviation. 

If you make a second mouse click while holding down the CTRL key on your keyboard,

above rule will also be applied on the gray value of the selected pixel.

Target |  The new mask that will be created or if you select an existing mask, Mimics will take into account the gray values of the pixels that are already in this mask.   
---|---  
Fill Cavities |  This feature enables filling internal gaps of the selected mask and places this in a new mask or in an existing one.  
Multiple Layer |  When this is marked, Mimics will look in the complete data set. When it is unmarked, the function will only be applied on the single slice.   
Seed point |  Shows the Hounsfield/Gray value of the last selected seed point.  
Deviation |  Min and max indicate the deviation range from the seed point.  
Distance | Limits the operation to a user defined cylinder.
