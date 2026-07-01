### Boolean Operations

![](Mimics Reference Guide/../Resources/Images/BooleanOperations.png)

The Boolean operations allow you to make all different kinds of combinations based on two masks. It is a very useful tool to reduce the work that needs to be done when separating two joints.

After entering the appropriate masks and operation, click on the Apply button.

The threshold limits of the resulting mask will be updated according to the values of the masks A and B and the operation applied: 

  * Subtraction (Minus):


Threshold value = Threshold value mask A

  * Intersection:   
lower threshold = max (low mask A, low mask B))


higher threshold = min (high mask A, high mask B))

  * Union:   
lower threshold = min (low mask A, low mask B))


higher threshold = max (high mask A, high mask B))

Example: Separation of a knee joint

The image with the green mask is the starting situation. As you see the tibia and femur are connected with each other.

So some erasing is necessary. The result is shown in the cyan mask. You only have to edit one part. Then region grow it e.g. to the purple mask. Now you can subtract the purple mask with the first structure from the cyan mask that contains both structures and you immediately have the second structure (red mask).

![](Mimics Reference Guide/../Resources/Images/knee%20boolean.png) ![](Mimics Reference Guide/../Resources/Images/Boolean%20operations_253x275.png)
