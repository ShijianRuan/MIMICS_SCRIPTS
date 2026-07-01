# Mask based

By using the material assignment from mask, you can use the segmentation in your project to assign materials to your elements. For each used mask, a linear material expression can be created. Each element is assigned to a mask based on the volume of intersection of that element with each mask. If an element has the same intersection volume with several masks, the mask with the highest priority is used for assigning a material to that element.

**Note** : If there are elements in the volume mesh that fall completely outside the selected mask, one extra material is created for those elements. This material is typically colored gray.

**Selecting the masks**

Click on the entity selection box to select which masks you want to use. Depending on the order of selection the priorities are set. For each mask a material tab is created. For each mask a separate material histogram and Material Editor is created.

**Number of materials**

Allows to discretize the gray values by dividing the range of gray values that occur in the mask into a specified number of equal sized intervals that each represents a material. The center gray value of each interval is chosen as a representative for that interval.

![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_Mask1_657x500.png) |  ![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_Mask2_640x500.png)  
---|---
