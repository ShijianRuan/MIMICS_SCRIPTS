####  Fit Centerline  
  
This tool allows you to calculate the centerline of a Part. These centerlines can be exported together with a range of measurements. 

To start the centerline calculation, first select the **Analyze** menu, and then, choose **Centerline > Fit Centerline**. The Fit centerline dialog will appear as shown below.

![](Mimics Reference Guide/../Resources/Images/fit_centerline_without_params_246x341.png)

In the fit centerline window, first choose the Part from the list. Next, choose the Smooth factor you would like to apply to your output centerline. The centerline smoothing deploys a Fourier smoothing method that removes high frequency noises from the initial centerline. The level of smoothing is determined by a threshold. The smoothing stops when at least one point in the branch deviates from its original position more than the threshold (mm) = (smooth factor) * (average branch radius).

Note: sometimes in case of very sharp turns in your geometry, the centerline may go outside your Part. In such cases, calculate the centerline again with a lower smoothing factor. 

The centerline extraction algorithm automatically detects the detail in your geometry and sets the values of two critical parameters, namely Resolving resolution and Distance between control points. A short description of these parameters is below: 

Resolving resolution |  This parameter indicates the size of the tubular structure which will be recognized by the algorithm, and of which the centerline will be extracted. A smaller resolving resolution (in mm) will allow the extraction of a centerline in tubular structures with a smaller diameter. A higher resolving resolution (in mm) will decrease the amount of small vessels, detail or noise recognized, and the creation of a centerline in these details.  
---|---  
Distance between control points |  This parameter sets the distance between two successive control points between which the centerline is interpolated. A smaller distance between control points (in mm) will allow the extraction of a centerline which follows more closely the geometry changes over the tubular structure. For tubular structures with a smaller diameter, a smaller distance between the control points should be used to capture all details.  
  
![](Mimics Reference Guide/../Resources/Images/fit_centerline_with_params_195x268.png)

If you wish to edit these parameters manually, you can expand the centerline extraction dialog by pressing the Show Params button. The expanded dialog is shown below. To access the Resolving resolution and Distance between control points parameters, switch the mode to Manual by pressing the corresponding check option. The resolving resolution can be adjusted according to the size of the tubular structures of which a centerline will be extracted.

In the automatic extraction of the centerline, the distance between the control points is variable, and adjusted to the size of the tubular structures of which a centerline will be extracted.Switching to the Fixed option gives you full control on the spacing between the control points. As a general guideline, if you are working with blood vessels, a large distance between consecutive control points will fail to detect very small vessels emerging out of the main vessel. Moreover, less detail of the geometry will be incorporated in the centerline. It is recommended to use the Variable option for good results. 

The resulting centerline consists of branch sets (one main branch set, and possibly several other branch sets). These branch sets consist of one or more branch segments confined by branching points and/or end points. The branch sets are ordered by the estimated average volume of the blood vessels. So, a shorter branch with a higher estimated average volume may precede a longer one.
