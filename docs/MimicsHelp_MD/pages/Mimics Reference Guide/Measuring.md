#### Measuring  
  
There are three measurement methods:

  * 4 point method (suggested for technical parts): The user will see four full lines numbered from 1 to 4, each with a different color. The full lines represent the input. The software calculates the intersection point of the vertical lines with the profile line: four points are calculated. The corresponding HU values can be read by following the full horizontal lines. On each side of the peak, a point is calculated as follows: P1 is at the position of the profile line where the HU value equals �T(gvalue2-gvalue1)+gvalue1� and lies between line1 and line2 on the horizontal axis. The dashed white lines indicate this point. The �T� function in above expression is the percentage of threshold difference. Most of the time this is about 50%. P2 is calculated in a similar way but with lines 3 and 4 as parameters. This point is indicated by the dashed yellow lines. The distance between P1 and P2 is the requested dimension.


  * 4 interval method (suggested for technical parts): Instead of indicating 4 points, the user will indicate 4 intervals. For each interval the average value is calculated. These four average values will then take over the role of the 4 points as described in the 4 point method.


  * Threshold method (suggested for medical images): Start the thresholding and drag the threshold line to the right position. Click on the End Thresholding button. The yellow and white dotted lines will move to the intersection between the profile line and the threshold line. The distance between the lines is displayed in mm.


The 4 point method and the 4 interval method are suggested for technical CT images, while the threshold method is suggested for medical applications.

In case you choose the 4 point method or 4 interval method, you will also see two horizontal lines (yellow and white). They indicate the position of the threshold value following the percentage filled out in the dialog box. 

The measurements are shown in the images and are saved to the project. They are also listed in the Measurements tab page of the project management. Like this you can always refer to the measurements made and pop-up exactly the same profile line as initially created.
