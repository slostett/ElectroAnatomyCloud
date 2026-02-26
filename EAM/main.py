import argparse
import ast
import sys
from display import *
from register import *
from graph import *

def parse_euler_transform(euler_str):
    """Parse and validate the --euler_transform input."""
    try:
        angles = ast.literal_eval(euler_str)
        if (not isinstance(angles, tuple)) or (len(angles) != 3):
            raise ValueError
        return angles
    except Exception:
        print(f"Error: --euler_transform must be a tuple of three numbers, e.g., --euler_transform \"(135, 0, 90)\"")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Mesh-CT Alignment Pipeline")
    parser.add_argument('--meshpath', type=str, required=True, help="Path to the .mesh file")
    parser.add_argument('--segpath', type=str, required=True, help="Path to the .nii.gz segmentation mask file")
    parser.add_argument('--xmlfolder', type=str, help="Path to folder containing voltage XMLs")
    parser.add_argument('--output_path', type=str, default=None, help="Optional path to save the output SimpleITK image (e.g., .nii.gz)")
    parser.add_argument('--plot', action='store_true', help="Flag to display the plot interactively")
    parser.add_argument('--segchannel', type=int, help="Input channel index for segmentation mask (e.g., 2)")
    parser.add_argument('--compute_euler_transform', action='store_true', help="Flag to check all rotations and find the best.")
    parser.add_argument('--euler_transform', type=parse_euler_transform, help="Known euler transform as a tuple, e.g., \"(135, 0, 90)\"")
    parser.add_argument('--kmeans_alignment', action='store_true', help="K-means based alignment optimization (for EAMs to LA)")

    print('Welcome to the ElectroAnatomyCloud EAM and CT alignment tool.')

    args = parser.parse_args()

    my_mesh = Mesh(meshpath=args.meshpath)
    la_seg_image = sitk.ReadImage(args.segpath)
    la_mask = extract_label_channel(la_seg_image, label_id=args.segchannel)
    la_shell = PointCloud(la_mask)

    if args.kmeans_alignment:
        my_mesh.cluster_points_kmeans(5)
        my_mesh.merge_n_closest_clusters(3)

    my_mesh.prealign(la_shell)

    if args.compute_euler_transform:
        vertices_aligned, transform, angles = euler_search_icp(fixed=la_shell.get_vertices(), moving=my_mesh.get_vertices())
        print('Applying computed euler transform:', transform)
        my_mesh.apply_transform(transform)
        # my_mesh.register_mesh_icp(la_shell.get_vertices()) TODO implement for PC

    if args.euler_transform:
        print('Applying user-specified euler transform:', args.euler_transform)
        my_mesh.apply_euler_transform(args.euler_transform)

    _, _, icp_cost = my_mesh.register_mesh_icp(la_shell)

    metrics = compute_alignment_metrics(my_mesh.get_vertices(), la_shell.get_vertices())
    print("\n=== Alignment Quality Report ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print("================================\n")

    mesh_img = my_mesh.mesh_to_sitk(la_mask)

    print('Plotting images:')
    plot_sitk_images(mesh_img, la_mask, name1='EAM Mesh', name2='Left Atrial Segmentation')

    if args.output_path:
        sitk.WriteImage(mesh_img, args.output_path)
        print(f"Saved aligned mesh image to: {args.output_path}")

if __name__ == "__main__":
    main()

'''
# Patient 1L — baseline run (prealign + ICP, no Euler search)
# Baseline scores (2026-02-23):
#   mean_surface_dist_mm:   ~2.6
#   symmetric_mean_dist_mm: ~24.0
#   rms_dist_mm:            ~3.3
#   hausdorff_95pct_mm:     ~111

python main.py `
 --meshpath "C:/Users/steph/Documents/UNC Cardiac Imaging/EAM data/1L/Patient 2025_07_08/PVI/Export_PVI-08_19_2025-15-41-50/2-LA FAM.mesh" `
 --segpath "C:/Users/steph/Documents/UNC Cardiac Imaging/CT Data/final nii/0000403E.nii.gz" `
 --segchannel 2 `
 --output_path "C:/Users/steph/Documents/UNC Cardiac Imaging/results/test_baseline.nii.gz" `
 --kmeans_alignment
'''