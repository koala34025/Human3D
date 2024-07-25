# Human3D Minimum Mask3D Model Inference Codes
Reference: https://github.com/human-3d/Human3D/issues/14#issuecomment-2242811047

One python file to conduct inference on egobody dataset (and also your custom data) with pretrained Mask3D model (idk if Human3D model works or not, might need some more tweaks).

The point clouds output of this python file has been checked to be the same with the output from the original evaluation script (./scripts/eval/eval_mask3d.sh).

See `myscript.sh` for the usage of the one python file and you can find more details in the git log. Hope it helps!
