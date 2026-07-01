#### Create Primitives

The Create Primitives tool allows you to create analytical primitives with a variety of creation methods.

![](Mimics Reference Guide/../Resources/Images/create_primitives_dialog_1.png)

  * First select the **type** of object that you want to create. The tool supports creation of points, lines, planes, cylinders, circles, spheres and splines.
  * After having selected the type of object to create, the list of creation **methods** will automatically adapt.
  * After having selected the creation method, the **input parameters** to be supplied will automatically adapt. For instance, when you selected to create a Line with the method �Through 2 points�, input fields for supplying two points will appear.


![](Mimics Reference Guide/../Resources/Images/create_primitives_dialog_2.png)

An input point can be selected in three possible ways. You can indicate a point in a 2D view or 3D view (left click to indicate the point), select an already existing analytical point via the dropdown control with the icon ![](Mimics Reference Guide/../Resources/Images/create_primitives_dropdown.png), or select an already existing analytical point by clicking on it in the Project Management tab. When hovering your mouse over the 2D or 3D views, existing points will highlight in blue when hovering over them, as an indication that they can be selected. After having selected a point, it will be visualized in green in the 2D and 3D views (until �Create� is pressed). You can then proceed with selecting the next point.

Selection of other types of inputs (such as lines etc) in other creation methods works in the same way.

  * After having selected all inputs, a preview of the object to be created automatically appears, and you can still revise any inputs. Finally you can change the **name** and **color** of the object to be created. Pressing �Create� will create the object. The dialog window remains open, but all input fields in the dialog will be cleared. You can then immediately proceed with creating the next primitive or press �Close�.


One particular creation method is �Fit to surface�. This method is available for Line, Sphere, Plane and Cylinder.

![](Mimics Reference Guide/../Resources/Images/create_primitives_dialog_3.png)

As �Fitting entity�, select a part. By default, the object to be created will be fitted to the entire fitting entity. Alternatively, you can fit to only a particular area of the fitting entity. Press the �Mark� button and a marking toolbar will appear.

![](Mimics Reference Guide/../Resources/Images/create_primitives_mark_dialog.png)

You can then mark the area of interest on the fitting entity. To mark, hold the left mouse button. Pressing SHIFT while marking will �mark through� the entity. Pressing CTRL while holding the left mouse button will unmark.

![](Mimics Reference Guide/../Resources/Images/create_primitives_example.png)

**Type** |  **Method** | Description  
---|---|---  
Point |  Center of gravity | Creates a point at the center of gravity of a part  
|  Closest point | Creates the closest point to a given point on a given entity. Supported types of entity are parts, lines and planes (when the �Entity� field in the dialog is in focus and the user hovers over objects in the 2D or 3D views, objects that are supported as input entity are highlighted).   
| Coordinates | Creates a point with the supplied (X,Y,Z) coordinates. Note that the coordinates can only be entered via keyboard input. It is possible to copy-paste coordinates in text format into the input fields of the dialog with a single copy-paste action. This works of the coordinates are separated by space, tab or semicolon, e.g. 1.0 2.0 3.0. If you wish to create a point at a certain position in the 2D or 3D views, please use Draw Point instead of Create Primitives.   
| Line and Plane intersection | Creates a point at the intersection of a line and plane. Exceptions: if the line is within the plane or the line and plane don't intersect, a warning is returned.   
| Lines intersection | Creates a point at the intersection of two lines. Exceptions: if the lines coincide or don't intersect, a warning is returned.   
| Midpoint | Creates the midpoint between two given points  
| Project point on part | Projects a point an a part along a given direction. If �Project through� is unchecked, it will create (at most) one point, namely the closed point to the given point. If �Project through� is checked, it will create multiple points (not only the closed one).   
|  |   
Line | Fit to surface | Fits a line to either a part or marked triangles on the part. See general explanation above.  
| Inertia axes | Creates three lines corresponding to the inertia axes of the given part.  
| Origin, direction and length | Creates a line starting from a given point in a certain direction and a given length.  
| Planes intersection | Creates a line at the intersection of two planes. Exceptions: if the planes coincide or don't intersect, a warning is returned.   
| Through 2 points | Creates a line through 2 points  
|  |   
Circle | Center, normal and radius | Creates a circle with a given center and radius, in a plane with a given normal.  
| Through 3 points | Creates a circle through 3 points  
|  |   
Sphere | Center and radius | Creates a sphere with a given center and radius.  
| Fit to surface | Fits a sphere to either a part or marked triangles on the part. See general explanation above.  
| Through 4 points | Creates a sphere through 4 points.  
|  |   
Plane | Fit to surface | Fits a plane to either a part or marked triangles on the part. See general explanation above.  
| Origin and normal | Creates a plane with a given original an normal.  
| Through 3 points | Creates a plane through 3 points.  
|  |   
Cylinder | 2 points and radius | Creates a cylinder with a given radius and an axis going through 2 given points.  
| Fit to surface | Fits a cylinder to either a part or marked triangles on the part. See general explanation above.  
| Through 3 points | Creates a cylinder based on 3 given points. The first and second point will form the begin and end point of the cylinder axis. The third point will determines the radius of the cylinder (the radius is equal to the distance between the third point and the infinite cylinder axis).  
|  |   
Spline | Project on plane | Creates a spline as a projection of a given spine on a plane (direction of projection is normal to the plane)
