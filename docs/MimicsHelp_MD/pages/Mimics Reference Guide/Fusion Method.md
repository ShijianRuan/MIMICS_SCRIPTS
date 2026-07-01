#### Fusion Method

You can select different Fusion Methods from the list:

Add |  GVr = GV1 + GV2 If GVr is higher than 4095GV, it is set to 4095.  
---|---  
Subtract |  GVr = GV1 - GV2 If GVr is lower than 0GV, it is set to 0.  
Multiply |  GVr = GV1 x GV2 If GVr is higher than 4095GV, it is set to 4095.  
Divide |  GVr = GV1 / GV2 GV1 is divided by GV2.  
Difference |  GVr = Abs(GV1 - GV2) The absolute value is taken of the difference between GV1 and GV2.  
Average |  GVr = (GV1 + GV2) / 2 The average is taken of GV1 and GV2.  
Min |  GVr = Minimum(GV1,GV2) The minimum is taken of GV1 and GV2.  
Max |  GVr = Maximum(GV1,GV2) The maximum is taken of GV1 and GV2.  
AND |  GVr = GV1 AND GV2 A bit wise AND is done of GV1 and GV2.  
OR |  GVr = GV1 OR GV2 A bit wise OR is done of GV1 and GV2.  
XOR |  GVr = GV1 XOR GV2 A bit wise XOR is done of GV1 and GV2.  
Transparent |  GVr = GV1 x (GV2 / 4095) The grayvalue of Dataset 2 is determining the transparency of the voxels of Dataset 1.  
Opaque |  GVr = GV2 The grayvalues of Dataset 2 are used. The result is that Dataset 2 is transformed to the coordinate system of Dataset 1.  
  
![](Mimics Reference Guide/../Resources/Images/AirCompensation.PNG) | The Air Compensation option allows achieving desirable results by removing the black spaces with actual gray values when using fusion method other than Max.  
---|---  
  
It is recommended to use Max fusion method along with air compensation when stitching two datasets; when registering two datasets not using the Z-axis, it is recommended to switch OFF air compensation.
