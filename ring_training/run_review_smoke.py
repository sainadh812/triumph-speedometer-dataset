"""Run the real 512px twenty-image learning check and record honest diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import torch
from torch.utils.data import DataLoader

from ring_data import RingDataset
from ring_train import RingTrainConfig, train, load_model
from ring_eval import evaluate, save_evaluation, comparison_plot, prediction_sheet

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'rotation_dataset/audits/20261017_100006_266720_f43a79f6_results/extracted/best_model.pt'
SOURCE_SHA = 'ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--mode', default='transfer')
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--finetune-epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--stop-after-epochs', type=int, default=None)
    parser.add_argument('--experiment', default='review20_learning_v1')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected verified rotation epoch-19 weights')
    data = ROOT / 'ring_geometry_pilot/review20'
    experiment = Path(__file__).resolve().parent / 'runs' / args.experiment
    cfg = RingTrainConfig(data_dir=str(data), output_dir=str(experiment), batch_size=args.batch_size,
                          head_epochs=args.head_epochs, finetune_epochs=args.finetune_epochs,
                          device=args.device, review_smoke=True, amp=args.device != 'cpu')
    result = train(cfg, mode=args.mode, source_checkpoint=str(SOURCE) if args.mode == 'transfer' else None,
                   search_root=str(experiment), stop_after_epochs=args.stop_after_epochs)
    dataset = RingDataset(data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    run = Path(result['run_dir'])
    before_model, _ = load_model(result['source_baseline_checkpoint_path'], device=args.device)
    before, before_rows = evaluate(before_model, loader, args.device)
    save_evaluation(run, 'before', before, before_rows)
    del before_model
    after_model, _ = load_model(result['best_model_path'], device=args.device)
    after, after_rows = evaluate(after_model, loader, args.device)
    save_evaluation(run, 'after', after, after_rows)
    comparison_plot(before, after, run / 'learning_check.png')
    prediction_sheet(after_model, dataset, args.device, run / 'model_predictions.png')
    result['detailed_before'] = before
    result['detailed_after'] = after
    result['comparison_plot'] = str(run / 'learning_check.png')
    result['prediction_sheet'] = str(run / 'model_predictions.png')
    (run / 'review_smoke_report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (experiment / 'latest_review_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'run_dir': str(run), 'best_epoch': result['best_epoch'],
                      'before': before, 'after': after}, indent=2))


if __name__ == '__main__':
    main()
