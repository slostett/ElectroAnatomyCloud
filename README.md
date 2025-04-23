A package for processing data of electroanatomical mappings from the Carto3D system and aligning them with binary segmentations of MRI or CT.

Inputs should be .mesh filetypes, and for voltage data, should be a series of .xmls, one for each point in the .mesh. The system will attempt to automatically find and label these points with either bipolar or unipolar voltages.

The package offers a suite of prealignment tools, including center of mass matching and PCA with long axis matching. Alignment happens in the point cloud space. 

Once aligned, the file can be returned to a sitk via the function mesh_to_sitk. 

As of now, the files are only available directly here. I'm currently working on getting the package available via pip and useable via command line.
