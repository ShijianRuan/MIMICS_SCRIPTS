# Analyze Airways

The **Analyze Airways** tool compares the bronchi volume and surface area of two airways. The comparison is realized by matching the corresponding branch points, subsequently the airway trees are made topology corresponding and the endings are pruned to the same length. Finally a volume analysis measurement is created in the measurement tab and visualized in the 3D view.

**General workflow**

  1. Initialization
  2. Centerline Matching
  3. Airway Cutting
  4. Volume Analysis


**Initialization**

**Analyze Airways** requires two labelled centerlines as input.

To import a centerline from another project use the Copy and Paste functionality between two Mimics instances or select Import Project under the **File** menu.

From the dropdown menu select the centerlines you want to compare.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways.png)

**Centerline Matching**

**Airway Matching** connects the corresponding branch points and endings. The connections between matched branch points are visualized by green connection lines; the matched endings are visualized by orange connections lines.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways2.png)

**Options**

Browsing Mode | Allows visualizing and updating the labels.  
---|---  
Editing Mode | Allows to add, remove or update connections.  
3D visibility | Shows / hides the 3D models.  
  
**Editing mode**

**Edit connection between branch points**

To change a branch point connection, grab the endpoint of the connection and drag it to a neighboring branching point.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways3.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways4.png)  
---|---  
To move a branch connection point, hold the Left Mouse Button and drag the connection point to its new position. | Release the Left Mouse Button to apply the change.  
  
**Remove a connection**

To remove a connection, select the connection and select Remove Connection from the context menu.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways5.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways6.png)  
---|---  
Left-click on the connection to select it. | From the context menu select **Remove Connection**.  
  
**Add a new connection**

To add a branch point connection, select the branch points, Right-Click on the branch connection point and select Add Connection from the context menu. To create the connection, select a branching point on the other centerline.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways7.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways8.png)  
---|---  
Left-Click on a branch point to select it, Right-click and select **Add Connection** from the context menu. | Select the corresponding branch point on the other centerline.  
  
**Browsing mode**

**Verify Label**

To verify the labels on the two centerlines, select a connection in Browsing Mode, a list will give you an overview of the branch labels that constitute the branch point.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways9.png)  
---  
Select a connection in browsing mode.  
  
**Update Label**

By making the branch labels consistent for the two centerlines, you can improve the matching results.

To update the branch label, Double-Click on the branch and select the correct label from the dropdown menu in the **Change Branch Label** dialog.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways10.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways11.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways12.png)  
---|---|---  
Double-Click launches the Change Branch Label Dialog. | Select from the dropdown list a new label. | Centerline matching will be updated based on the new label.  
  
**Airway Cutting**

Based on the matching results, the **Airway Cutting** generates the cutting planes at b-levels and at endpoints. After cutting the branch sections will be topology corresponding and pruned at the same length, allowing volume and surface comparisons between branch sections.

Cutting of the branches will succeed for the green colored planes. Cutting will not succeed for fuchsia colored cutting planes. The position and orientation of those cutting planes can be manually updated. In cases no optimal cutting plane can be found manually the branch will not be cut.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways13.png)

**Options**

Reset | Resets the cutting planes position or orientation  
---|---  
Save CFD model | Saves a non-manifold model of the cut Airways  
3D visibility | Shows / hides the 3D models   
  
**Adjust the cutting plane**

To update the position of the cutting plane, grab the center of cutting plane and dragging it over the centerline.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways14.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways15.png)  
---|---  
Left-click on the center of the cutting plane and hold the left mouse button. | Drag the center of the cutting plane over the centerline.  
  
To update the orientation of the cutting plane, grab the contour of the cutting plane and move mouse.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways16.png) |  ![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways17.png)  
---|---  
Left-click on the contour of the cutting plane and hold the left mouse button. |  Move the mouse while holding the left-mouse button to change the cutting plane orientation.  
  
_Volume Analysis_

The volume comparison is visualized in the 3D view. To make changes in a previous step, click **Back** or select the relevant step from the left panel.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways18.png)

The measurements are stored in the measurement tab and can be exported via the **Txt** export in the **Export** menu.

![](Mimics Reference Guide/Pulmonary/../../Resources/Images/AnalyseAirways19.png)
