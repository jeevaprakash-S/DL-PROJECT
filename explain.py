"""Attention and gradient×input explanations for one held-out window."""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, torch
import config
from dataset import make_loaders
from model import build_model

def load_model():
    checkpoint = torch.load(config.CHECKPOINT_DIR / "world_model_v2_best.pt", map_location=config.DEVICE, weights_only=False)
    model = build_model(checkpoint["num_features"]); model.load_state_dict(checkpoint["model"]); model.eval(); return model

def explain(index=0, top_k=15):
    loaders, _, _ = make_loaders(); dataset = loaders["test"].dataset
    if not 0 <= index < len(dataset): raise IndexError(f"index must be 0..{len(dataset)-1}")
    names = (config.PROCESSED_DATA_DIR / "feature_cols.txt").read_text().splitlines(); x, y_next, truth = dataset[index]; x = x.unsqueeze(0).to(config.DEVICE).requires_grad_(True)
    model = load_model(); forecast, logits, layers = model(x, return_attention=True); probabilities = torch.softmax(logits, 1)[0]; prediction = int(probabilities.argmax())
    model.zero_grad(); logits[0, prediction].backward(); importance = (x.grad[0].abs() * x.detach()[0].abs()).sum(0).cpu().numpy()
    attention = torch.stack(layers).mean(0)[0, -1].detach().cpu().numpy()
    ranking = np.argsort(importance)[::-1][:top_k]
    result = {"index": index, "true_stage": config.STAGES[int(truth)], "predicted_stage": config.STAGES[prediction], "confidence": float(probabilities[prediction].detach().cpu()), "stage_probabilities": dict(zip(config.STAGES, map(float, probabilities.detach().cpu().numpy()))), "forecast_error": float((forecast[0].detach().cpu()-y_next).square().mean()), "top_features_gradient_x_input": [{"feature": names[i], "score": float(importance[i])} for i in ranking], "temporal_attention": attention.tolist()}
    (config.OUTPUT_DIR / f"explanation_seq{index}.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4)); ax.bar(range(1, len(attention)+1), attention); ax.set(xlabel="input timestep (oldest → newest)", ylabel="attention", title="Temporal attention into final world state"); fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / f"attention_seq{index}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(range(len(ranking)), importance[ranking][::-1]); ax.set_yticks(range(len(ranking)), [names[i] for i in ranking][::-1]); ax.set(xlabel="gradient × input", title="Features influencing predicted stage"); fig.tight_layout(); fig.savefig(config.OUTPUT_DIR / f"feature_attribution_seq{index}.png", dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--index", type=int, default=0); parser.add_argument("--top-k", type=int, default=15); args = parser.parse_args(); explain(args.index, args.top_k)
