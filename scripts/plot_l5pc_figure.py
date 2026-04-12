from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec


class RunBundle:
    def __init__(self, label: str, summary: dict, predictions: dict[str, np.ndarray]) -> None:
        self.label = label
        self.summary = summary
        self.predictions = predictions


def _resolve_summary(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        candidate = path / "l5pc_summary.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find l5pc_summary.json in {path}")
        return candidate
    if path.suffix == ".json":
        return path
    raise ValueError(f"Expected a run directory or summary json path, got {path}")


def load_run(path_str: str) -> RunBundle:
    summary_path = _resolve_summary(path_str)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dump_path = Path(summary["prediction_dump"])
    if not dump_path.is_absolute():
        dump_path = summary_path.parent / dump_path
    predictions_npz = np.load(dump_path)
    predictions = {key: predictions_npz[key] for key in predictions_npz.files}
    label = summary.get("run_name", summary_path.parent.name)
    return RunBundle(label=label, summary=summary, predictions=predictions)


def plot_task_panel(ax, runs: list[RunBundle]) -> None:
    ax.axis("off")
    labels = "\n".join(f"- {run.label}" for run in runs)
    text = (
        "Question\n"
        "Does dendrite-soma factorization improve parameter efficiency and fidelity\n"
        "for L5PC biophysical neuron emulation?\n\n"
        "Included runs\n"
        f"{labels}\n\n"
        "Metrics\n"
        "Voltage RMSE, spike AUC, low-FPR pAUC, spike-time cross-correlation, params"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11)
    ax.set_title("Panel A. Task And Model Setup", loc="left", fontsize=12, fontweight="bold")


def plot_trace_panel(fig, outer_spec, run: RunBundle) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_spec, height_ratios=[1, 2], hspace=0.08)
    raster_ax = fig.add_subplot(inner[0])
    trace_ax = fig.add_subplot(inner[1], sharex=raster_ax)

    inputs = run.predictions.get("representative_input")
    pred_voltage = run.predictions.get("representative_pred_voltage")
    label_voltage = run.predictions.get("representative_label_voltage")
    if inputs is None or inputs.size == 0:
        raster_ax.axis("off")
        trace_ax.axis("off")
        return

    start = 500 if inputs.shape[0] > 1000 else 0
    stop = min(start + 2000, inputs.shape[0])
    raster = inputs[start:stop].T
    raster_ax.imshow(raster, aspect="auto", interpolation="nearest", cmap="Greys")
    raster_ax.set_ylabel("Synapse")
    raster_ax.set_title(f"Panel B. Representative Raster And Voltage Trace ({run.label})", loc="left", fontsize=12, fontweight="bold")

    trace_ax.plot(label_voltage[start:stop], color="black", linewidth=1.2, label="Ground truth")
    trace_ax.plot(pred_voltage[start:stop], color="tab:orange", linewidth=1.0, linestyle="--", label="Prediction")
    trace_ax.set_xlabel("Time (ms)")
    trace_ax.set_ylabel("Voltage (mV)")
    trace_ax.legend(frameon=False, loc="upper right")


def plot_pareto_panel(ax, runs: list[RunBundle]) -> None:
    params = np.asarray([run.summary["param_count"] for run in runs], dtype=np.float32)
    rmse = np.asarray([run.summary["voltage_rmse"] for run in runs], dtype=np.float32)
    auc = np.asarray([run.summary["spike_auc"] for run in runs], dtype=np.float32)
    scatter = ax.scatter(params, rmse, c=auc, cmap="viridis", s=90)
    for run, x, y in zip(runs, params, rmse, strict=True):
        ax.annotate(run.label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel("Voltage RMSE (mV)")
    ax.set_title("Panel C. Pareto Efficiency", loc="left", fontsize=12, fontweight="bold")
    colorbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spike AUC")


def plot_roc_alignment_panel(ax, runs: list[RunBundle]) -> None:
    for run in runs:
        fpr = run.predictions["roc_fpr"]
        tpr = run.predictions["roc_tpr"]
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{run.label} (AUC={run.summary['spike_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Panel D. ROC And Spike-Timing Alignment", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    inset = ax.inset_axes([0.58, 0.1, 0.38, 0.38])
    for run in runs:
        inset.plot(
            run.predictions["cross_corr_lags"],
            run.predictions["cross_corr_curve"],
            linewidth=1.2,
            label=run.label,
        )
    inset.axvline(0, color="0.7", linestyle=":", linewidth=1.0)
    inset.set_xlabel("Lag (ms)", fontsize=8)
    inset.set_ylabel("Corr", fontsize=8)
    inset.tick_params(labelsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a draft L5PC figure from one or more run outputs.")
    parser.add_argument("runs", nargs="+", help="Run directories or l5pc_summary.json files.")
    parser.add_argument("--output", default="outputs/l5pc_figure.png", help="Output image path.")
    args = parser.parse_args()

    runs = [load_run(path_str) for path_str in args.runs]
    runs.sort(key=lambda run: (run.summary["param_count"], run.summary["voltage_rmse"]))
    trace_run = min(runs, key=lambda run: run.summary["voltage_rmse"])

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    outer = gridspec.GridSpec(2, 2, figure=fig)

    ax_a = fig.add_subplot(outer[0, 0])
    plot_task_panel(ax_a, runs)
    plot_trace_panel(fig, outer[0, 1], trace_run)
    ax_c = fig.add_subplot(outer[1, 0])
    plot_pareto_panel(ax_c, runs)
    ax_d = fig.add_subplot(outer[1, 1])
    plot_roc_alignment_panel(ax_d, runs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
