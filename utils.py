import os
from subprocess import call
import subprocess

def download_model(download_url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

    model_name = download_url.split("/")[-1]
    save_path = os.path.join(folder, model_name)

    if os.path.exists(save_path):
        return

    print(f"Downloading {model_name}...")
    subprocess.run(["wget", "-O", save_path, download_url], check=True)

    print(f"{model_name} downloaded and saved to {folder}.")



model_dl_urls = {
    "lineart": "https://huggingface.co/ListAngel/control_any3/blob/main/control_any3_canny.pth",
}

annotator_dl_urls = {
    "body_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/body_pose_model.pth",
    "dpt_hybrid-midas-501f0c75.pt": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt",
    "hand_pose_model.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/hand_pose_model.pth",
    "mlsd_large_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth",
    "mlsd_tiny_512_fp32.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_tiny_512_fp32.pth",
    "network-bsds500.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/network-bsds500.pth",
    "upernet_global_small.pth": "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/upernet_global_small.pth",
}


# def download_model(model_name, urls_map):
#     """
#     Download model from huggingface with wget and save to models directory
#     """
#     model_url = urls_map[model_name]
#     relative_path_to_model = model_url.replace("https://huggingface.co/lllyasviel/ControlNet/resolve/main/", "")
#     if not os.path.exists(relative_path_to_model):
#         print(f"Downloading {model_name}...")
#         call(["wget", "-O", relative_path_to_model, model_url])

def get_state_dict_path(model_name):
    """
    Get path to model state dict
    """
    return f"./models/control_sd15_{model_name}.pth"