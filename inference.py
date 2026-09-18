"""Stream held-out windows through the trained model and emit early warnings."""
import argparse, json
import numpy as np, torch
import config
from dataset import make_loaders
from model import build_model

def load_model():
    checkpoint = torch.load(config.CHECKPOINT_DIR / "world_model_v2_best.pt", map_location=config.DEVICE, weights_only=False)
    model = build_model(checkpoint["num_features"]); model.load_state_dict(checkpoint["model"]); model.eval()
    thresholds = json.loads((config.OUTPUT_DIR / "alert_threshold.json").read_text())
    return model, thresholds

def stream(limit=200):
    loaders, _, _ = make_loaders(); ds = loaders["test"].dataset; model, thresholds = load_model(); alerts = []
    for index in range(min(len(ds), limit if limit >= 0 else len(ds))):
        x, y_next, actual = ds[index]
        with torch.no_grad(): forecast, logits = model(x.unsqueeze(0).to(config.DEVICE)); probs = torch.softmax(logits, 1)[0].cpu().numpy()
        stage_idx, error = int(probs.argmax()), float((forecast[0].cpu() - y_next).square().mean())
        is_stage_alert = stage_idx != 0 and probs[stage_idx] >= thresholds["stage_probability_threshold"]
        is_novelty_alert = error >= thresholds["forecast_error_threshold"]
        if is_stage_alert or is_novelty_alert:
            alerts.append({"sequence_index": index, "predicted_stage": config.STAGES[stage_idx], "confidence": float(probs[stage_idx]), "true_stage": config.STAGES[int(actual)], "forecast_error": error, "stage_alert": bool(is_stage_alert), "novelty_alert": bool(is_novelty_alert)})
    (config.OUTPUT_DIR / "alerts_log.json").write_text(json.dumps(alerts, indent=2))
    print(f"{len(alerts)} alerts from {min(len(ds), limit if limit >= 0 else len(ds))} held-out windows; saved outputs/alerts_log.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--limit", type=int, default=200); stream(parser.parse_args().limit)
