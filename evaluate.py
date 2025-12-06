from pathlib import Path
import torch

from src.dataset import build_dataloader, load_yaml
from src.models.network import SeisMambaKAN
from src.metrics import evaluate_model_on_loader

def main():
    main_cfg = load_yaml("configs/config.yaml")
    model_cfg = load_yaml("configs/model_config.yaml")
    paths_cfg = load_yaml("configs/paths.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model + load checkpoint
    model = SeisMambaKAN(model_cfg).to(device)
    state = torch.load("experiments/exp_005/checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(state)

    # Build val loader
    val_loader = build_dataloader(
        split="val",
        cfg=main_cfg,
        paths_cfg=paths_cfg,
        is_train=False,
    )

    exp_dir = Path("experiments/exp_005")
    metrics = evaluate_model_on_loader(
        model=model,
        data_loader=val_loader,
        device=device,
        main_cfg=main_cfg,
        split_name="val",
        exp_dir=exp_dir,
    )

    print(metrics)

if __name__ == "__main__":
    main()
