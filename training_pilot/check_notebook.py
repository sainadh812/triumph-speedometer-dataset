"""Execute every notebook cell in a short local CPU software check.

Use a Python environment containing torch, torchvision, nbclient and ipykernel.
The default Colab pilot remains GPU/512/pretrained; only this local test uses
128 pixels, random weights and 8/4 samples. No performance conclusion is drawn.
"""
from pathlib import Path
import json
import os
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CHECKS = ROOT / "checks"
CHECKS.mkdir(exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="speedometer_notebook_check_"))
kernel_root = work / "kernels"
kernel_dir = kernel_root / "speedometer-smoke"
kernel_dir.mkdir(parents=True)
launcher = work / "launch_kernel.py"
# Loading torch before ipykernel avoids a Windows DLL initialization-order
# conflict in this local environment. This does not change notebook code.
launcher.write_text("import torch\nfrom ipykernel.kernelapp import IPKernelApp\nIPKernelApp.launch_instance()\n", encoding="utf-8")
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv": [sys.executable, str(launcher), "-f", "{connection_file}"],
    "display_name": "Speedometer isolated CPU test", "language": "python",
}), encoding="utf-8")
specs = KernelSpecManager(kernel_dirs=[str(kernel_root)], ensure_native_kernel=False)
manager = KernelManager(kernel_name="speedometer-smoke", kernel_spec_manager=specs)
notebook = nbformat.read(PROJECT / "Speedometer_Training_Colab.ipynb", as_version=4)
environment = os.environ.copy()
environment.update({
    "SPEEDOMETER_RUN_MODE": "smoke", "SPEEDOMETER_WORK_DIR": str(work),
    "SPEEDOMETER_DATASET_ZIP": str(PROJECT / "speedometer_bw500.zip"),
    "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
})
client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                        resources={"metadata": {"path": str(work)}})
started = time.perf_counter()
report = {"passed": False, "scope": "Local execution smoke test only; no model performance claim",
          "python": sys.executable, "temporary_work_dir": str(work)}
try:
    client.execute(env=environment)
    outputs = list((work / "smoke_").glob("*/experiment_results.json"))
    assert len(outputs) == 1
    experiments = json.loads(outputs[0].read_text())
    result = experiments[0]
    assert result["run_mode"] == "smoke" and result["training_samples"] == 8 and result["validation_samples"] == 4
    assert result["epochs_run"] == 2 and not result["pretrained"] and result["config"]["input_size"] == 128
    for key in ("checkpoint_path", "metrics_path", "predictions_path", "worst_predictions_path", "learning_curves_path"):
        assert Path(result[key]).is_file(), key
    archives = list((work / "smoke_").glob("*_results.zip"))
    assert len(archives) == 1
    report.update({"passed": True, "code_cells_executed": sum(c.cell_type == "code" for c in notebook.cells),
                   "training_samples": 8, "validation_samples": 4, "epochs": 2,
                   "input_size": 128, "pretrained": False, "checkpoint_reload_and_inference": "passed",
                   "result_archive": str(archives[0]), "colab_gpu_execution": "not_run"})
except Exception as error:
    report["error"] = str(error)
    raise
finally:
    report["elapsed_seconds"] = time.perf_counter() - started
    nbformat.write(notebook, CHECKS / "smoke_executed.ipynb")
    (CHECKS / "notebook_smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
