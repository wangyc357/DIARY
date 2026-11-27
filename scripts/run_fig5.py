"""Repeat the current noise-strategy experiment and preserve its results."""

import argparse
import csv
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def accuracy_rows(log_text, run_number):
    """Read per-round MA/BA from INFO lines, including grouped runs."""
    stage = 'training'
    round_number = 0
    main_accuracy = None
    rows = []
    for line in log_text.splitlines():
        if re.search(r'_+Begin Recover(?: Group \d+)?_+', line):
            stage = 'recovery'
            round_number = 0
        if ('Grouped global model testing accuracy:' in line or
                'Recovered grouped global model testing accuracy:' in line):
            continue
        match = re.search(r'(?:Testing|testing) accuracy:\s*([\d.]+)%', line)
        if match:
            main_accuracy = float(match.group(1))
            continue
        match = re.search(r'(?:Attack|attack) accuracy:\s*([\d.]+)%', line)
        if match and main_accuracy is not None:
            round_number += 1
            rows.append((run_number, stage, round_number,
                         main_accuracy, float(match.group(1))))
            main_accuracy = None
    return rows


def save_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/test/fmnist.yaml')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output-dir', default='results/fig5')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be at least 1')

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    with config_path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    dataset = config['FL']['dataset']
    recovery = config['FL']['recover']
    dp_path = root / 'config/dp' / f"{config['FL']['dp']}.yaml"
    with dp_path.open(encoding='utf-8') as stream:
        dp_config = yaml.safe_load(stream)['Fed_dp']
    dp_config = config.get('Fed_dp', dp_config)
    with (root / 'config/recover' / f'{recovery}.yaml').open(encoding='utf-8') as stream:
        recover_config = yaml.safe_load(stream)['Recover']['args']
    recover_config = config.get('Recover', {}).get('args', recover_config)
    method = dp_config['noise_config']['type'].upper()
    if method == 'IPLN' and recover_config['recalculate_noise']:
        method = 'DIPLN'
    source_log = root / 'logs' / dataset / f'{recovery}.log'
    output_root = (root / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_dir = output_root / f'{dataset}_{method}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    result_dir.mkdir()

    all_rows = []
    for run_number in range(1, args.runs + 1):
        console_log = result_dir / f'run_{run_number}_console.log'
        print(f'[{run_number}/{args.runs}] {dataset}-{method}: running...', flush=True)
        with console_log.open('w', encoding='utf-8') as stream:
            process = subprocess.run(
                [sys.executable, 'main.py', '--config', str(config_path)],
                cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False,
            )
        if source_log.is_file():
            saved_log = result_dir / f'run_{run_number}.log'
            shutil.copy2(source_log, saved_log)
        if process.returncode:
            raise RuntimeError(
                f'Run {run_number} failed (exit {process.returncode}); see {console_log}'
            )
        if not source_log.is_file():
            raise FileNotFoundError(f'Run {run_number} did not create {source_log}')
        log_text = saved_log.read_text(encoding='utf-8')
        rows = accuracy_rows(log_text, run_number)
        if not rows:
            rows = accuracy_rows(console_log.read_text(encoding='utf-8'), run_number)
        if not rows:
            raise RuntimeError(f'Run {run_number} has no MA/BA pairs; see {saved_log}')
        all_rows.extend(rows)
        print(f'[{run_number}/{args.runs}] saved {len(rows)} points', flush=True)

    save_csv(result_dir / 'per_run_accuracy.csv',
             ['run', 'stage', 'round', 'main_accuracy', 'backdoor_accuracy'], all_rows)
    groups = {}
    for _, stage, round_number, main_acc, backdoor_acc in all_rows:
        group = groups.setdefault((stage, round_number), ([], []))
        group[0].append(main_acc)
        group[1].append(backdoor_acc)
    mean_rows = []
    for (stage, round_number), (main_values, backdoor_values) in sorted(groups.items()):
        mean_rows.append((stage, round_number, len(main_values),
                          statistics.fmean(main_values),
                          statistics.stdev(main_values) if len(main_values) > 1 else 0,
                          statistics.fmean(backdoor_values),
                          statistics.stdev(backdoor_values) if len(backdoor_values) > 1 else 0))
    save_csv(result_dir / 'mean_accuracy.csv',
             ['stage', 'round', 'runs', 'mean_main_accuracy', 'std_main_accuracy',
              'mean_backdoor_accuracy', 'std_backdoor_accuracy'], mean_rows)
    print(f'Finished: {result_dir}', flush=True)


if __name__ == '__main__':
    main()
