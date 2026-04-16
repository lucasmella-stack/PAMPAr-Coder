import torch

for e in ["pampar_v3", "no_llaves", "single_stream", "vanilla_gpt"]:
    ckpt = torch.load(
        f"ablation_results/{e}/checkpoint.pt", map_location="cpu", weights_only=False
    )
    print(f"{e}: paso={ckpt['paso']}")
