# Starting Pulmonary  
  
After registration of the Pulmonary module the tools will appear in the menu and in the pulmonary toolbar.

**On the menu bar:**

  * The Pulmonary menu lists **Segment Airway** and manual segmentation editing tools, a tool to prepare the airway for CFD analysis, namely **Cut Airways** , a tool to compare the volume and surface area of the lower airway branches, namely **Analyse Airways**. Finally the menu contains automated tools to segment the lung and lung lobes, namely **Segment Lung**.


![](Mimics Reference Guide/Pulmonary/../../Resources/Images/PulmonaryModule.png)

**In the Project management:**

  * No new tabs will appear. The created objects will appear in the following existing tabs:


**Tool** | **Project Management Tab** | **Object**  
---|---|---  
Segment Airway |  Masks Parts |  Airway mask Airway Part  
Threshold Region Grow | Masks | New mask/Existing mask  
Dynamic Region Grow | Masks | New mask/Existing mask  
Label Airway | Analysis Objects | Centerline  
Analyse Airway | Measurements |   
Cut Airways | Parts | <Centerline name>_Cut  
Segment Lungs and Lobes |  Masks 3D model |  Left Lung, Right Lung Right Upper Lobe Right Middle Lobe Right Lower Lobe Left Upper Lobe Left Lower Lobe  
  
Note: If you're not able to start the pulmonary module, it is possible that you haven't entered the License File yet. Go to **Help > Licenses** and select a key file.
