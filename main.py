import argparse
from pathlib import Path
from src.config import Config
from src.utils import ensure_dirs


def main():
    p = argparse.ArgumentParser(description="DeepLeafAnalyzer complete pipeline")
    p.add_argument("--mode", choices=["save_config", "train_unet", "make_roi", "train_classifier", "evaluate", "infer"], required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--image", default=None)
    p.add_argument("--weights", default=None)
    args = p.parse_args()
    cfg = Config.load(args.config) if args.config else Config()
    ensure_dirs(cfg.checkpoint_dir, cfg.result_dir, cfg.figure_dir, cfg.roi_dir)
    if args.mode == "save_config":
        cfg.save("config.json"); print("Saved config.json")
    elif args.mode == "train_unet":
        from src.train_unet import train_unet; train_unet(cfg)
    elif args.mode == "make_roi":
        from src.roi_fusion import generate_roi_dataset
        generate_roi_dataset(cfg.raw_dir, cfg.roi_dir, str(Path(cfg.checkpoint_dir)/"best_unet.pth"), args.weights, cfg.img_size, cfg.device)
    elif args.mode == "train_classifier":
        from src.train_classifier import train_classifier; train_classifier(cfg)
    elif args.mode == "evaluate":
        from src.evaluate import evaluate_classifier; evaluate_classifier(cfg, args.weights)
    elif args.mode == "infer":
        from src.inference import predict_image
        weights = args.weights or str(Path(cfg.checkpoint_dir)/"best_hybridleaf.pth")
        print(predict_image(args.image, weights, cfg))

if __name__ == "__main__": main()
