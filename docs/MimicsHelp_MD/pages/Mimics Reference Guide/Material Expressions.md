#### Material Expressions

**Density Expression**

The density expression can be used for translating the gray value of an element to a density.

To do this, an empirical expression of the form ρ = α+β*GV can be entered to convert the gray value into a density value. This method can also be used to export gray values: if ρ = 0+1*GV is used as a density expression, the representatives of each interval are written out as densities.

**Note** : Most FEA software do not allow you to enter a density with a negative value, so make sure you choose your expression accordingly or adjust the values manually in the material editor.

**E-Modulus expression**

Based on the density value, an expression can be entered to define the e-modulus for each material. The entered expression will only be used if the checkmark is enabled. If there is no density value available for a certain element, the e-modulus value will also remain empty.

**Poisson expression**

Based on the density value, an expression can be entered to define the poisson coefficient for each material. The entered expression will only be used if the checkmark is enabled. If there is no density value available for a certain element, the poisson coefficient value will also remain empty.

**Note** : About units (Hounsfield units/Gray values): Hounsfield units are the unit image data that comes from medical scanning devices. Grayvalues are the unit that is used internally in Mimics. Both units relate as value in GV = value in HU + 1024. Mimics has a preference setting to select which unit is used in the user interface (Edit -> Preferences -> General -> Pixel Unit).

**Warning** : The following default values are used when leaving one of the checkboxes on OFF. Density = 0, Young�s modulus = 0 and Poisson Coefficient = 0.5.
