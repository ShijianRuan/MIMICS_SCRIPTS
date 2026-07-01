### Work with multiple image sets  
  
**Link an object to an image set**

A link is a physical connection between an object and an image set.

You can access the links in the Project Management tabs under the name Images.

![](Mimics Reference Guide/../Resources/Images/ObjectLinking_320x298.png)

An object can be linked to an Image set. Objects can also stay without a link. To modify a link click on the dropdown list that appears under the column Images. You can select the desired image set from the full list of image sets loaded in the project. 

Objects can only be linked to image sets and not to other objects.

**Image sets and Mimics tools**

By default the Mimics tools work with the currently active image set. The Mimics tools that operate on image sets have effect only on the currently active image set. For example in case you want to reslice an image set, if you follow the procedure that is described in the Reslice menu, only the currently active image set will be resliced. To reslice more than one image sets you need to follow the process as many times as the number of the image sets that you target. 

_Note_ : To apply any operation that has effect on an image set, you need to make this image set active first. 

**Objects in Mimics tools**

As mentioned in the section above, a Mimics object can be linked to an image set or it can be not linked to any of the image sets. A lot of tools of Mimics have a dropdown list or a selection table where you can choose objects to apply the tool. By default, only the objects that are linked to the currently active image set and the objects that are not linked to any image set appear when you work with a tool. 

In case the tool creates new objects, they will be linked to the currently active image set if the object supports the link to an image set. For details see the sections of the different Mimics objects.
