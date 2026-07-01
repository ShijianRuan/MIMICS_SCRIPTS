#### Import/export of objects between Mimics and Matlab scripts

You can use the above mentioned objects defined in your Mimics Project by adding them to the input section of the Run Matlab script dialog. Press the Add button in the Input section to add an input variable to your script. Input objects from Mimics can also be defined directly in the script by adding the command �%Mimics.input:� followed by the Variable name you would like to give to the Mimics object. Now an input variable has been defined. Next, you will need to assign a Mimics object to each of the input variables that were added. This can be done by simply chosing the object of interest from the drop down list in the Object column as shown below. 

![](Mimics Reference Guide/../Resources/Images/Mimics%2015.0%20Reference%20Guide_track_v2_no_numbers/030000A5_554x510.png)

Next, the output variables need to be defined by pressing the Add button in the Output column. You can also define the variables that you would like to outputs from your script to Mimics by adding �%Mimics.Output� command in the script. Note: that Mimics can only import objects defined in the correct formats into the project. Not all the variables used in the Matlab script will be imported into Mimics project. 

Once the input and output objects are set, simply press the Run script button to run your script. If the values of the output variables in your script are defined in the correct format, they can be imported to the project by clicking on �Import� button or �Import all� button.

Be sure to visit and contribute to our online repository of sample Matlab scripts at [http://uc.materialise.com/mimics/matlab](<http://uc.materialise.com/mimics/matlab>).
