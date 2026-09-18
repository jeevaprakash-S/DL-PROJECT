"""Train Version 2 with balanced sampling and class-balanced focal loss."""
import argparse, json, random, time
import numpy as np, torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import config
from dataset import make_loaders
from model import build_model

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)

def class_balanced_alpha(counts):
    # Effective-number weights are less explosive than inverse frequency.
    effective = 1.0 - np.power(config.EFFECTIVE_NUMBER_BETA, np.maximum(counts, 1))
    alpha = (1.0 - config.EFFECTIVE_NUMBER_BETA) / effective
    alpha = alpha / alpha.mean()
    return torch.tensor(np.minimum(alpha, config.MAX_ALPHA), dtype=torch.float32, device=config.DEVICE)

def focal_loss(logits, target, alpha):
    log_probs = torch.log_softmax(logits, 1); log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
    pt = log_pt.exp(); return (-alpha[target] * (1 - pt).pow(config.FOCAL_GAMMA) * log_pt).mean()

def forecast_error(pred, target): return (pred - target).square().mean(1)

def epoch(model, loader, optimizer, alpha=None):
    training = optimizer is not None; model.train(training)
    losses, fcs, clss, pred_all, label_all, errors = [], [], [], [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y_next, y_stage in loader:
            x, y_next, y_stage = x.to(config.DEVICE), y_next.to(config.DEVICE), y_stage.to(config.DEVICE)
            if training: optimizer.zero_grad(set_to_none=True)
            forecast, logits = model(x); fc = forecast_error(forecast, y_next).mean(); cls = focal_loss(logits, y_stage, alpha) if training else torch.nn.functional.cross_entropy(logits, y_stage)
            loss = config.FORECAST_LOSS_WEIGHT * fc + config.CLASS_LOSS_WEIGHT * cls
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses.append(loss.item()); fcs.append(fc.item()); clss.append(cls.item()); pred_all.append(logits.argmax(1).cpu().numpy()); label_all.append(y_stage.cpu().numpy()); errors.append(forecast_error(forecast, y_next).detach().cpu().numpy())
    labels, preds = np.concatenate(label_all), np.concatenate(pred_all)
    all_stage_f1 = f1_score(labels, preds, labels=range(config.NUM_STAGES), average="macro", zero_division=0)
    primary_f1 = f1_score(labels, preds, labels=config.PRIMARY_EVALUATION_INDICES, average="macro", zero_division=0)
    return {"loss": float(np.mean(losses)), "forecast_loss": float(np.mean(fcs)), "class_loss": float(np.mean(clss)), "macro_f1": float(all_stage_f1), "primary_macro_f1": float(primary_f1), "labels": labels, "preds": preds, "errors": np.concatenate(errors)}

def main(epochs=None, sanity=False, train_steps=None):
    config.ensure_dirs(); seed_everything(config.SEED)
    if sanity: train_steps = config.SANITY_TRAIN_STEPS
    elif train_steps is None: train_steps = config.FULL_TRAIN_STEPS
    loaders, features, counts = make_loaders(train_steps=train_steps)
    alpha = class_balanced_alpha(counts); print("focal alpha:", dict(zip(config.STAGES, np.round(alpha.cpu().numpy(), 3).tolist())))
    model = build_model(features); optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best, stale, history = -1.0, 0, []
    ckpt_path = config.CHECKPOINT_DIR / "world_model_v2_best.pt"
    for number in range(1, (epochs or config.EPOCHS) + 1):
        started = time.time(); train_metrics = epoch(model, loaders["train"], optimizer, alpha); val_metrics = epoch(model, loaders["val"], None)
        scheduler.step(val_metrics["primary_macro_f1"]); history.append({"epoch": number, **{f"train_{k}": v for k,v in train_metrics.items() if k not in ("labels","preds","errors")}, **{f"val_{k}": v for k,v in val_metrics.items() if k not in ("labels","preds","errors")}})
        print(f"epoch {number:02d}: {time.time()-started:.1f}s train_all_f1={train_metrics['macro_f1']:.3f} val_primary_f1={val_metrics['primary_macro_f1']:.3f} val_all_f1={val_metrics['macro_f1']:.3f} val_loss={val_metrics['loss']:.3f}")
        # Primary macro-F1 is the checkpoint criterion. Delivery and
        # Exploitation remain in training/prediction, but have too little
        # independent validation support for stable checkpoint selection.
        if val_metrics["primary_macro_f1"] > best + 1e-6:
            best, stale = val_metrics["primary_macro_f1"], 0
            torch.save({"model": model.state_dict(), "num_features": features, "epoch": number, "val_primary_macro_f1": best, "stages": config.STAGES}, ckpt_path)
        else:
            stale += 1
            if stale >= config.EARLY_STOPPING_PATIENCE: print("early stopping on primary validation macro-F1"); break
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False); model.load_state_dict(checkpoint["model"])
    val = epoch(model, loaders["val"], None); test = epoch(model, loaders["test"], None)
    report = classification_report(test["labels"], test["preds"], labels=range(config.NUM_STAGES), target_names=config.STAGES, zero_division=0)
    matrix = confusion_matrix(test["labels"], test["preds"], labels=range(config.NUM_STAGES)).tolist()
    test_support = np.bincount(test["labels"], minlength=config.NUM_STAGES)
    experimental = {stage: int(test_support[i]) for i, stage in enumerate(config.STAGES) if test_support[i] < config.MIN_RELIABLE_TEST_SUPPORT}
    evaluation_scope = {"model_training_stages": list(config.STAGES), "primary_evaluation_stages": list(config.PRIMARY_EVALUATION_STAGES), "minimum_reliable_test_support": config.MIN_RELIABLE_TEST_SUPPORT, "test_support": dict(zip(config.STAGES, map(int, test_support))), "experimental_stages_insufficient_test_support": experimental, "primary_test_macro_f1": test["primary_macro_f1"], "all_stage_exploratory_macro_f1": test["macro_f1"]}
    scope_note = "\n\nPRIMARY EVALUATION\n" + f"Stages: {', '.join(config.PRIMARY_EVALUATION_STAGES)}\n" + f"Primary test macro-F1: {test['primary_macro_f1']:.4f}\n" + f"All-stage exploratory macro-F1: {test['macro_f1']:.4f}\n" + f"Experimental stages with <{config.MIN_RELIABLE_TEST_SUPPORT} test windows: {experimental}\n"
    benign = val["errors"][val["labels"] == config.STAGE_TO_IDX["Benign"]]
    novelty_threshold = float(np.percentile(benign if len(benign) else val["errors"], config.ALERT_FORECAST_ERROR_PERCENTILE))
    (config.OUTPUT_DIR / "training_history.json").write_text(json.dumps(history, indent=2)); (config.OUTPUT_DIR / "test_classification_report.txt").write_text(report + "\nConfusion matrix:\n" + json.dumps(matrix) + scope_note); (config.OUTPUT_DIR / "evaluation_scope.json").write_text(json.dumps(evaluation_scope, indent=2))
    (config.OUTPUT_DIR / "alert_threshold.json").write_text(json.dumps({"forecast_error_threshold": novelty_threshold, "source": "p95 benign validation forecast error", "stage_probability_threshold": config.ALERT_STAGE_PROB_THRESHOLD}, indent=2))
    print(f"\nPRIMARY TEST macro-F1={test['primary_macro_f1']:.4f} ({', '.join(config.PRIMARY_EVALUATION_STAGES)})\nALL-STAGE exploratory macro-F1={test['macro_f1']:.4f}\n{report}{scope_note}\nNovelty threshold={novelty_threshold:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int); parser.add_argument("--sanity", action="store_true"); parser.add_argument("--train-steps", type=int)
    args = parser.parse_args(); main(args.epochs, args.sanity, args.train_steps)
