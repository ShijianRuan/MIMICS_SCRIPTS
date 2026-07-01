### Dynamic Region Grow

![](Mimics Reference Guide/../Resources/Images/DynamicRegionGrowing.png)

The Dynamic Region Grow tool allows you to segment an object based on the connectivity of gray values in a certain gray value range. It allows an easy segmentation of blood vessels, nerves, ... in CT images.

The Dynamic Region Growing function is the only operation where you don�t have to threshold. A threshold value will be set automatically, the minimum and maximum value of the created mask will serve as threshold values. 

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
  
Note: The deviation parameter in previous versions of Mimics was expressed with 1 byte (0-255), while from Mimics 8.0 the complete gray value range of 12 bits (0-4095) is used so the deviation can be better fine-tuned. This means that you have to use a deviation that is much higher than in Mimics 7.3 to get the same results. 

Example:

Selection of the alveolar nerve with one mouse click:

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/0200010D_207x202.jpg)   
2D |  ![](Mimics Reference Guide/../Resources/Images/dynamic%20region%20growing.png)   
3D  
---|---
