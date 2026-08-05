import os
import torch
from argparse import ArgumentParser
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams

# Maximize PyTorch GPU compilation performance
torch.backends.cudnn.benchmark = True

def train_progressive(dataset, opt, pipe, testing_iterations):
    for attr, default in [("img_ext", ""), ("depths", None), ("train_test_exp", False)]:
        if not hasattr(dataset, attr):
            setattr(dataset, attr, default)

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=None, resolution_scales=[1.0, 2.0, 4.0, 8.0])
    gaussians.training_setup(opt)

    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    available_keys = list(scene.train_cameras.keys())
    print(f"\n[INFO] Loaded dataset successfully. Available resolution keys: {available_keys}")
    
    scale_8x = 8 if 8 in available_keys else (0.125 if 0.125 in available_keys else available_keys[0])
    scale_4x = 4 if 4 in available_keys else (0.25 if 0.25 in available_keys else available_keys[0])
    scale_2x = 2 if 2 in available_keys else (0.5 if 0.5 in available_keys else available_keys[0])

    current_scale = scale_8x
    
    print("[INFO] Caching camera stack in memory for maximum speed...")
    viewpoint_stack = scene.getTrainCameras(scale=current_scale)

    for iteration in range(first_iter + 1, opt.iterations + 1):
        if iteration == 7001:
            print(f"\n[MILESTONE REACHED - ITER {iteration}]: Swapping data pipelines to 4x resolution scale.")
            current_scale = scale_4x
            viewpoint_stack = scene.getTrainCameras(scale=current_scale)
        elif iteration == 15001:
            print(f"\n[MILESTONE REACHED - ITER {iteration}]: Swapping data pipelines to 2x high-fidelity scale.")
            current_scale = scale_2x
            viewpoint_stack = scene.getTrainCameras(scale=current_scale)

        if iteration % 500 == 0 or iteration == 1:
            loss_val = Loss.item() if "Loss" in locals() else 0.0
            print(f"Iteration {iteration}/{opt.iterations} | Active Gaussians: {len(gaussians.get_xyz)} | Loss: {loss_val:.4f}")

        viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)]

        render_pkg = render(viewpoint_cam, gausians, pipe, background) if "gausians" in globals() else render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Loss = l1_loss(image, gt_image)
        Loss.backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(max_grad=opt.densify_grad_threshold, min_opacity=0.01, extent=scene.cameras_extent, max_screen_size=size_threshold, radii=radii)

            gaussians.optimizer.step()
            # SPEED TWEAK: Using set_to_none=True to bypass writing zeros across VRAM memory slots
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in testing_iterations:
            print(f"[SAVING CHECKPOINT]: Writing model state path at iteration {iteration}")
            scene.save(iteration)

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args()
    
    train_progressive(lp.extract(args), op.extract(args), pp.extract(args), [7000, 15000, 30000])
