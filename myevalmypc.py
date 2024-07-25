import hydra
import torch
import sys
import time
import os

from utils.utils import (
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
)

class InstanceSegmentation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = hydra.utils.instantiate(cfg.model)


    def forward(
        self,
        x,
        point2segment=None,
        raw_coordinates=None,
        is_eval=True,
        clip_feat=None,
        clip_pos=None,
    ):
        x = self.model(
            x,
            point2segment,
            raw_coordinates=raw_coordinates,
            is_eval=is_eval,
            clip_feat=clip_feat,
            clip_pos=clip_pos,
        )
        return x
    

from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import initialize, compose

# imports for input loading
import albumentations as A
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d

def get_model(checkpoint_path=None):


    # Initialize the directory with config files
    with initialize(config_path="conf"):
        # Compose a configuration
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    cfg.general.checkpoint = checkpoint_path

    # would be nicd to avoid this hardcoding below
    cfg.general.experiment_name = "Human3D_eval"
    cfg.general.project_name = "human3d"
    cfg.general.num_targets = 16
    cfg.data.num_labels = 16
    # cfg.model = "mask3d_hp"
    # cfg.loss = "set_criterion_hp"
    # cfg.model.num_human_queries = 5
    # cfg.model.num_parts_per_human_queries = 16
    cfg.model.num_queries = 5

    cfg.trainer.check_val_every_n_epoch = 1
    cfg.general.topk_per_image = -1  # Use -1 to indicate no limit or a special behavior
    cfg.model.non_parametric_queries = False
    cfg.trainer.max_epochs = 36
    cfg.data.batch_size = 4
    cfg.data.num_workers = 10
    cfg.general.reps_per_epoch = 1
    cfg.model.config.backbone._target_ = "models.Res16UNet18B"

    cfg.data.part2human= True
    cfg.loss.num_classes=2
    cfg.model.num_classes=2
    cfg.callbacks="callbacks_instance_segmentation_human"

    cfg.general.checkpoint = checkpoint_path
    cfg.general.train_mode = False
    cfg.general.save_visualizations = True

        
        #TODO: this has to be fixed and discussed with Jonas
        # cfg.model.scene_min = -3.
        # cfg.model.scene_max = 3.

    # # Initialize the Hydra context
    # hydra.core.global_hydra.GlobalHydra.instance().clear()
    # hydra.initialize(config_path="conf")

    # Load the configuration
    # cfg = hydra.compose(config_name="config_base_instance_segmentation.yaml")

    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model
        )
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return model


def load_mesh(pcl_file):
    
    # load point cloud
    input_mesh_path = pcl_file
    # armadillo_mesh = o3d.data.ArmadilloMesh()
    # mesh = o3d.io.read_triangle_mesh(armadillo_mesh.path)
    # o3d.visualization.draw_geometries([mesh])

    # mesh = o3d.io.read_triangle_mesh(input_mesh_path)
    mesh = o3d.io.read_point_cloud(input_mesh_path)
    # o3d.visualization.draw_geometries([mesh])


    points = np.asarray(mesh.points)
    colors = np.asarray(mesh.colors)
    # for cropping
    # min_bound = np.array([1.27819920, -3.04697800, -1.14556611])
    # max_bound = np.array([4.38896847, 1.98770964, 1.54433572])

    # # Apply cropping
    # in_bounds = (points >= min_bound) & (points <= max_bound)
    # in_bounds = in_bounds.all(axis=1)
    # points = points[in_bounds]
    # colors = colors[in_bounds]
    # mesh.vertices = o3d.utility.Vector3dVector(points)
    # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh

def prepare_data(mesh, device):
    
    # normalization for point cloud features
    # wtf?
    # color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    # color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)

    color_mean, color_std = np.array([1.0, 1.0, 1.0]), np.array(
                [1.0, 1.0, 1.0]
            )
    # how about this?
    #?cfg.data.validation_dataset.color_mean_std = [(0.5, 0.5, 0.5), (1, 1, 1)]
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    
    points = np.asarray(mesh.points)

    if len(mesh.colors) == 0:
        # Default color - white
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors = (np.asarray(mesh.colors) * 255).astype(np.uint8)
    
    # fix rotation bug
    points = points[:, [0, 2, 1]]
    points[:, 2] = -points[:, 2]
    # print(points)

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :] # (this belongs to body part?)
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"]) # so should be sample[1]

    # exmaple needs these, idk why?
    colors = np.hstack((colors, points))
    colors[:, :3] = 1.0  # make sure no color information is leaked
    
    # print("~~~~~~~~~~~~~~~")
    # print(colors)
    # print("~~~~~~~~~~~~~~~")

    voxel_size = 0.02
    coords = np.floor(points / voxel_size) # points = sample[0]
    
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=torch.from_numpy(coords).contiguous(),
        features=colors, # should send in sample[1]
        return_index=True,
        return_inverse=True,
    )

    # print("---------------------------------")
    # print(coords)
    # print("--------------")
    # print(unique_map)
    # print("--------------")
    # print(colors) #   SAME
    # print("---------------------------------")

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, features = ME.utils.sparse_collate(coords=coordinates, feats=features)
    # features = torch.cat(features, dim=0) # idk why comment
    raw_coordinates = features[:, -3:]
    features = features[:, :-3]

    # print(coordinates) # SAME
    # print(features)

    # print("^^^^^^^^^^^^^^^^^^^^^^^^")
    # print(coordinates)
    # print(features)
    # print("^^^^^^^^^^^^^^^^^^^^^^^^")
    
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )
    
    # same inverse_map
    return data, points, colors, features, unique_map, inverse_map, raw_coordinates


def map_output_to_pointcloud(mesh, 
                             outputs, 
                             inverse_map,
                             full_res_coords, 
                             label_space='scannet200',
                             confidence_threshold=0.8):
    # print(outputs)
    # parse predictions
    logits = outputs["pred_logits"]
    masks = outputs["pred_masks"]
    # print(logits)
    # print(masks)
    # reformat predictions
    logits = torch.functional.F.softmax(logits, dim=-1)[..., :-1]

    logits = logits[0].detach().cpu()
    masks = masks[0].detach().cpu()
    # print("First masks=======================================")
    # print(masks)
    # print(masks.shape)

    result_pred_mask = (masks > 0).float()
    # print("Second masks=======================================")
    # print(masks)
    # print(masks.shape)

    # solve logits
    scores_per_query, labels_per_query = logits.max(dim=1)
    heatmap = masks.float().sigmoid()
    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
        result_pred_mask.sum(0) + 1e-6
    )
    score = scores_per_query * mask_scores_per_image
    # print(score)

    masks = result_pred_mask.detach().cpu()[inverse_map].numpy()  # full res
    # print("Thrid masks=======================================")
    # print(masks)
    # print(masks.shape) # unsorted mask
    # print(score)

    sort_scores = score.sort(descending=True)
    sort_scores_index = sort_scores.indices.cpu().numpy()
    sort_scores_values = sort_scores.values.cpu().numpy()

    # print(sort_scores_index)
    sorted_masks = masks[:, sort_scores_index]
    # print(sorted_masks)
    classes = labels_per_query
    sort_classes = classes[sort_scores_index]

    pred_coords = []
    for i in reversed(range(sorted_masks.shape[1])):
        if sort_scores_values[i] > 0.5:
            print(i)
            mask_coords = full_res_coords[
                sorted_masks[:, i].astype(bool), :
            ]

            label = sort_classes[i]

            if len(mask_coords) == 0:
                continue

            pred_coords.append(mask_coords)
            print(mask_coords.shape)

    return pred_coords


def save_colorized_mesh(points, pred_coords, output_file):
    # Define a simple color map for classes
    color_map = {
        0: [0, 0, 0],       # Black for background
        1: [255, 0, 0],     # Red for first human
        2: [0, 255, 0],      # Green for second human
        3: [0, 0, 255]
    }

    # Create a numpy array to hold the colors for each point
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    
    # Mark all points as background initially
    colors[:] = color_map[0]
    
    # really bad time complexity
    # Color points for the first human
    for coord in pred_coords[0]:
        idx = np.where((points == coord).all(axis=1))[0]
        if idx.size > 0:
            colors[idx[0]] = color_map[1]

    # Color points for the second human
    for coord in pred_coords[1]:
        idx = np.where((points == coord).all(axis=1))[0]
        if idx.size > 0:
            colors[idx[0]] = color_map[2]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1] for Open3D

    # Save the point cloud with color information
    o3d.io.write_point_cloud(output_file, pcd)


def visualize_mesh(mesh_file):
    # Load the colorized mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    
    # Ensure the mesh is correctly loaded and has vertex colors
    if not mesh.has_vertex_colors():
        raise ValueError("The mesh does not have vertex colors.")
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])


def visualize_pc(mesh_file):
    mesh = o3d.io.read_point_cloud(mesh_file)
    
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh])


def log_times(log_file, step_name, elapsed_time):
    with open(log_file, 'a') as f:
        f.write(f'{step_name}: {elapsed_time:.2f} seconds\n')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: python myevalmypc.py [.ply file path]')
        sys.exit(1)

    pointcloud_file = sys.argv[1]
    log_file = f'myevaloutput/performance/{os.path.basename(pointcloud_file).replace(".ply", ".log")}'

    # Measure time for model loading
    start_time = time.time()
    model = get_model('./checkpoints/mask3d.ckpt')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log_times(log_file, 'Model loading time', time.time() - start_time)

    # Load input data
    start_time = time.time()
    mesh = load_mesh(pointcloud_file)
    log_times(log_file, 'Mesh loading time', time.time() - start_time)

    # Prepare data
    start_time = time.time()
    data, points, colors, features, unique_map, inverse_map, raw_coordinates = prepare_data(mesh, device)
    log_times(log_file, 'Data preparation time', time.time() - start_time)

    # Run model
    start_time = time.time()
    with torch.no_grad():
        outputs = model(data, raw_coordinates=raw_coordinates)
    log_times(log_file, 'Model inference time', time.time() - start_time)

    # Map output to point cloud
    start_time = time.time()
    pred_coords = map_output_to_pointcloud(mesh, outputs, inverse_map, points)
    log_times(log_file, 'Output mapping time', time.time() - start_time)

    # Save colorized mesh
    start_time = time.time()
    output_path = 'myevaloutput/my_pc.ply'
    save_colorized_mesh(points, pred_coords, output_path)
    log_times(log_file, 'Mesh saving time', time.time() - start_time)

    # Visualize mesh
    # start_time = time.time()
    visualize_pc(output_path)
    # log_times(log_file, 'Mesh visualization time', time.time() - start_time)
