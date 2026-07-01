# Gray value based

This method allows splitting up to gray value range in multiple material types. To each type a different material expression can be assigned.

**Limit to Mask**

This option allows for a recalculation of the gray values per volume element limiting it to the values covered by the selected mask. This to intercepts the deviation in the boundary elements due to the partial volume effect. As boundary voxels typically represent multiple tissues by excluding these voxels, the material assignment will become more accurate.

**Note** : If there are elements in the volume mesh that fall completely outside the selected mask, one extra material is created for those elements. This material is typically colored gray.

**Material type**

When adding a number of additional material types, the gray value range will be split equally by the corresponding amount of types. Each Material type can have its own material expressions allowing the user to create piecewise linear material properties. The gray value range of each material type can be adjusted. The ranges of the neighboring types will be adjusted automatically, to keep the whole range covered. For each material type a separate Material Editor view is created. The + and x button allow to easily add and remove material types.

**Note** : When working with multiple material types you can change the gray value range by moving the separator in the histogram.

**Number of materials**

Allows to discretize the gray values of a material type by dividing the range of gray values that occur in the volume mesh into a specified number of equal sized intervals that each represents a material. The center gray value of each interval is chosen as a representative for that interval.

![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_GrayValue1_638x500.png) |  ![](Mimics Reference Guide/../Resources/Images/FEACFD_AssignMaterial_GrayValue2_637x500.png)  
---|---
