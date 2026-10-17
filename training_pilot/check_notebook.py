"""Execute the actual notebook twice against mock Drive: fresh then resume.

Only local CPU random-weight software checks run: 128px, 8/4 samples, two epochs.
This does not estimate model accuracy or exercise Google authorization/remote IO.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import time
import zipfile

import torch
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CHECKS = ROOT / "checks"
CHECKS.mkdir(exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="speedometer_drive_notebook_check_"))
mock_drive = work / "mock_drive" / "SpeedometerTraining"
kernel_root = work / "kernels"
kernel_dir = kernel_root / "speedometer-smoke"
kernel_dir.mkdir(parents=True)
launcher = work / "launch_kernel.py"
# Load torch before ipykernel to avoid local Windows DLL initialization ordering.
launcher.write_text("import torch\nfrom ipykernel.kernelapp import IPKernelApp\nIPKernelApp.launch_instance()\n", encoding="utf-8")
(kernel_dir / "kernel.json").write_text(json.dumps({
    "argv":[sys.executable,str(launcher),"-f","{connection_file}"],
    "display_name":"Speedometer isolated CPU test","language":"python",
}),encoding="utf-8")


def run_notebook(mode):
    specs = KernelSpecManager(kernel_dirs=[str(kernel_root)],ensure_native_kernel=False)
    manager = KernelManager(kernel_name="speedometer-smoke",kernel_spec_manager=specs)
    notebook = nbformat.read(PROJECT / "Speedometer_Training_Colab.ipynb",as_version=4)
    environment = os.environ.copy()
    environment.update({
        "SPEEDOMETER_RUN_MODE":"smoke", "SPEEDOMETER_TRAIN_MODE":mode,
        "SPEEDOMETER_WORK_DIR":str(work), "SPEEDOMETER_DRIVE_ROOT":str(mock_drive),
        "SPEEDOMETER_EXPERIMENT":"smoke_persistence_v1",
        "SPEEDOMETER_DATASET_ZIP":str(PROJECT / "speedometer_bw500.zip"),
        "MPLBACKEND":"Agg", "OMP_NUM_THREADS":"4", "MKL_NUM_THREADS":"4",
    })
    client = NotebookClient(notebook,km=manager,timeout=300,allow_errors=False,
                            resources={"metadata":{"path":str(work)}})
    try:
        client.execute(env=environment)
    finally:
        nbformat.write(notebook,CHECKS / f"drive_smoke_{mode}_executed.ipynb")
    return sum(cell.cell_type=="code" for cell in notebook.cells)


def read_result(path):
    results = json.loads(path.read_text(encoding="utf-8"))
    assert len(results)==1
    result=results[0]
    assert result["run_mode"]=="smoke" and result["training_samples"]==8 and result["validation_samples"]==4
    assert result["epochs_run"]==2 and result["target_epoch"]==2
    assert not result["pretrained"] and result["config"]["input_size"]==128
    for key in ("checkpoint_path","latest_checkpoint_path","metrics_path","predictions_path","worst_predictions_path","learning_curves_path"):
        assert Path(result[key]).is_file(),key
        assert Path(result[key]).is_relative_to(mock_drive),key
    return result


started=time.perf_counter()
report={"passed":False,"scope":"Local mock-Drive execution only; no accuracy claim",
        "python":sys.executable,"temporary_work_dir":str(work)}
try:
    code_cells=run_notebook("fresh")
    experiment=mock_drive / "experiments" / "smoke_persistence_v1"
    first_paths=list(experiment.glob("runs/*/experiment_results.json"))
    assert len(first_paths)==1
    first=read_result(first_paths[0])
    assert first["epochs_this_call"]==2
    sidecars=list(experiment.glob("runs/*/epoch_*.complete.json"))
    assert len(sidecars)==2
    checkpoint_hashes={}
    required={"model_state_dict","optimizer_state_dict","scaler_state_dict","rng_state",
              "loader_generator_states","training_state","best_state","history","identity"}
    for sidecar in sidecars:
        manifest=json.loads(sidecar.read_text(encoding="utf-8"))
        checkpoint=sidecar.parent / manifest["checkpoint"]
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert digest==manifest["sha256"]
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert payload["format_version"]==2 and required.issubset(payload)
        assert payload["training_state"]["target_epoch"]==2
        assert payload["training_state"]["epoch"]==manifest["epoch"]
        checkpoint_hashes[str(checkpoint)]=digest
    code_cells+=run_notebook("resume_latest")
    all_paths=list(experiment.glob("runs/*/experiment_results.json"))
    assert len(all_paths)==2
    second=read_result(next(path for path in all_paths if path!=first_paths[0]))
    assert second["epochs_this_call"]==0
    assert second["resume_metadata"]["full_state_restored"]
    assert second["resume_metadata"]["optimizer_restored"]
    assert second["resume_metadata"]["source_epoch"]==2
    assert first["checkpoint_path"]!=second["checkpoint_path"]
    for path,digest in checkpoint_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    archives=list(experiment.glob("runs/*_results.zip"))
    assert len(archives)==2
    for archive in archives:
        with zipfile.ZipFile(archive) as zipped:
            names=zipped.namelist()
            assert "best_model.pt" in names and "drive_paths.json" in names
            assert "train_core.py" in names and "notebook_workflow.py" in names
            assert not any(Path(name).name.startswith("epoch_") or ".partial" in name for name in names)
            assert zipped.testzip() is None
    report.update({"passed":True,"code_cells_executed":code_cells,"notebook_invocations":2,
                   "training_samples":8,"validation_samples":4,"epochs_trained":2,
                   "epochs_on_completed_resume":0,"target_epoch_preserved":2,
                   "input_size":128,"pretrained":False,"full_state_epoch_files":2,
                   "original_checkpoint_hashes_unchanged":True,"compact_archives":len(archives),
                   "checkpoint_reload_and_inference":"passed","drive_remote_upload":"not_tested",
                   "colab_gpu_execution":"not_run"})
except Exception as error:
    report["error"]=str(error)
    raise
finally:
    report["elapsed_seconds"]=time.perf_counter()-started
    (CHECKS / "notebook_smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2),flush=True)
