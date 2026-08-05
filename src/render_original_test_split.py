import torch
from scene import Scene
import os
from gaussian_renderer import GaussianModel, render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import render as render_module
from render import render_set  # reuse the exact same rendering/saving logic

if __name__ == "__main__":
    parser = ArgumentParser(description="Render original test split against a non-eval-trained model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--img_ext", type=str, default='jpg')
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # the original test-split filenames from step 1's --eval split
    orig_names = sorted(os.listdir(os.path.join(args.source_path, "images")))
    test_names = set(n for i, n in enumerate(orig_names) if i % 8 == 0)

    safe_state(args.quiet)
    dataset = model.extract(args)
    dataset.upscale = args.upscale
    dataset.img_ext = args.img_ext

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        try:
            scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, upscale=(dataset.upscale, dataset.upscale), skip_train=False)
        except:
            dataset.img_ext = dataset.img_ext.upper()
            scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, upscale=(dataset.upscale, dataset.upscale), skip_train=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        all_cams = scene.getTrainCameras() + scene.getTestCameras()
        for cam in all_cams:
            cam.image_height = cam.image_height * dataset.upscale
            cam.image_width = cam.image_width * dataset.upscale
            cam.original_image = cam.hr_image.to(cam.data_device)

        filtered_cams = [c for c in all_cams if c.image_name in test_names]
        print(f"Matched {len(filtered_cams)} of {len(test_names)} expected original test cameras")

        pipeline_params = pipeline.extract(args)
        render_module.args = args
    render_set(dataset.model_path, "orig_test_split", scene.loaded_iter, filtered_cams, gaussians, pipeline_params, background, dataset.train_test_exp, False)
