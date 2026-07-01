##### Centerline post-processing  
  
The centerline Post processing dialog allows smoothing and filtering options on the selected centerline. 

![](Mimics Reference Guide/../Resources/Images/CL%20post%20processing.jpg)

###### Additional smoothing

The smoothing functionality smooths out high frequency noise in the centerline curve via a Fourier smoothing method. This can improve the accuracy of your centerline measurements, for example curvature measurements. The value of the slider can be set from 0 to 1, indicating low to high levels of smoothing. Loss in accuracy of the centerline may occur when using very high levels of smoothing.

The level of smoothing is determined by a threshold. The smoothing stops when at least one point in the branch deviates from its original position more than the threshold (mm) = (smooth factor) * (average branch radius).

###### Filter short branches

Sometimes very small branches or noise branches are also detected in the centerline detection stage. These branches can be removed by using the Filter short branches. The function works by removing all peripheral (connected at one end to the main centerline tree) or lose branches (not connected to the main centerline tree) whose length is below the specified filtering threshold. If you are not sure which value to choose for the filtering threshold, go to centerline properties and click on the branch you feel is noise, the branch will be highlighted in the properties dialog. Note down the length of this branch and use this as the threshold value.
