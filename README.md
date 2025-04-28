A package for processing data of electroanatomical mappings from the Carto3D system and aligning them with binary segmentations of MRI or CT.

Inputs should be .mesh filetypes, and for voltage data, should be a series of .xmls, one for each point in the .mesh. The system will attempt to automatically find and label these points with either bipolar or unipolar voltages. The package offers a suite of prealignment tools, including center of mass matching and PCA with long axis matching. Alignment happens in the point cloud space. Once aligned, the file can be returned to a sitk via the function mesh_to_sitk. 

Examples of the SimpleITK visualization and PointCloud class visualization.

SITK
![](https://github.com/slostett/ElectroAnatomyCloud/blob/main/sitk.png)
PointCloud
![](https://github.com/slostett/ElectroAnatomyCloud/blob/main/cloud.png)

I can't post any of the EAMs or alignment data here since it's PHI. The data above is from an open source CT I segmented using totalsegmentator. If anyone has open source EAM data, let me know and I will process and add it here.

Example Usage of the CLI (recommended for simple use cases):
```
py align_mesh.py `
 --meshpath "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/ExportData28_02_25 16_19_56/Patient 2025_02_28/AF/Export_AF-02_28_2025-16-01-43/6-1-sinus.mesh" `
 --segpath "C:/Users/steph/Downloads/Sorted_0_6_channel0.nii" `
 --output_path "C:/users/steph/Documents/UNC Cardiac Imaging/results/EAM.nii.gz" `
 --segchannel 2 `
 --euler_transform "(0, 0, 0)" `
 --kmeans_alignment `
 --plot
```

Euler transform is the rotation which gives the lowest prealignment score. This can be brute forced using --compute_euler_transform as an argument, but it will take awhile. Once you've computed this once, you can save the transform and reapply it next time without manually computing using --euler_transform and passing in the tuple in quotations as above.

Example Usage of Imported Package:
```
from display import Mesh

my_mesh = Mesh(meshpath="path/to/mesh.mesh")  # need meshpath= or the class will assume you are passing in raw points.
my_mesh.initialize_voltages("path/to/folder/with/xml")  # the code will attempt to find the xml voltages with the same name as your mesh.
mesh.plot()
```

As of now, the files are only available directly here. I'm currently working on getting the package available via pip.
