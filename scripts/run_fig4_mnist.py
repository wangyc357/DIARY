"""Run the MNIST or Fashion-MNIST part of Figure 4 repeatedly."""

import argparse
import ast
import csv
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/test/mnist.yaml')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--output-dir', default='results/fig4')
    return parser.parse_args()


def find_accuracy_array(log_text, label, expected_rounds):
    arrays = []
    for line in log_text.splitlines():
        marker = f'{label} Accuracy:'
        if marker not in line:
            continue
        values = ast.literal_eval(line.split(marker, 1)[1].strip())
        if len(values) == expected_rounds:
            arrays.append(values)

    if arrays:
        return arrays[0]

    # Some logger configurations omit DEBUG summary lines. In that case,
    # recover the same training curve from the per-round INFO messages.
    training_text = re.split(
        r'_+Begin Recover(?: Group \d+)?_+', log_text, maxsplit=1
    )[0]
    if label == 'Main':
        pattern = r'(?:Testing|testing) accuracy:\s*([0-9]+(?:\.[0-9]+)?)%'
    elif label == 'Attacker':
        pattern = r'(?:Attack|attack) accuracy:\s*([0-9]+(?:\.[0-9]+)?)%'
    else:
        raise ValueError(f'Unsupported accuracy label: {label}')

    values = [float(value) for value in re.findall(pattern, training_text)]
    if len(values) >= expected_rounds:
        return values[:expected_rounds]

    raise RuntimeError(
        f'Cannot find {expected_rounds} training values for {label} Accuracy. '
        f'Found {len(values)} per-round values and no matching summary array.'
    )


def write_per_run_csv(path, main_accuracies, attack_accuracies):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['run', 'round', 'main_accuracy', 'backdoor_accuracy'])
        for run_index, (main_values, attack_values) in enumerate(
                zip(main_accuracies, attack_accuracies), start=1):
            for round_index, (main_value, attack_value) in enumerate(
                    zip(main_values, attack_values), start=1):
                writer.writerow([run_index, round_index, main_value, attack_value])


def write_mean_csv(path, main_accuracies, attack_accuracies):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow([
            'round',
            'mean_main_accuracy',
            'std_main_accuracy',
            'mean_backdoor_accuracy',
            'std_backdoor_accuracy',
        ])
        for round_index, (main_values, attack_values) in enumerate(
                zip(zip(*main_accuracies), zip(*attack_accuracies)), start=1):
            main_values = list(main_values)
            attack_values = list(attack_values)
            writer.writerow([
                round_index,
                statistics.fmean(main_values),
                statistics.stdev(main_values) if len(main_values) > 1 else 0.0,
                statistics.fmean(attack_values),
                statistics.stdev(attack_values) if len(attack_values) > 1 else 0.0,
            ])


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')

    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    with config_path.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    fl_config = config['FL']
    dataset = fl_config['dataset']
    if dataset not in {'mnist', 'fmnist'}:
        raise ValueError(
            'This script supports MNIST and Fashion-MNIST, '
            f'but the config uses {dataset!r}.'
        )

    expected_rounds = fl_config['rounds']
    method = 'find' if fl_config['with_defense'] else 'dpfl'
    recover_name = fl_config.get('recover', 'DIARY')
    source_log = project_root / 'logs' / dataset / f'{recover_name}.log'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = (project_root / args.output_dir / f'{dataset}_{method}_{timestamp}').resolve()
    result_dir.mkdir(parents=True, exist_ok=False)

    main_accuracies = []
    attack_accuracies = []

    for run_index in range(1, args.runs + 1):
        print(
            f'[{run_index}/{args.runs}] Running {dataset.upper()}-{method.upper()}...',
            flush=True,
        )
        console_log = result_dir / f'run_{run_index}_console.log'
        with console_log.open('w', encoding='utf-8') as console_stream:
            subprocess.run(
                [sys.executable, 'main.py', '--config', str(config_path)],
                cwd=project_root,
                stdout=console_stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

        if not source_log.is_file():
            raise FileNotFoundError(f'Expected training log was not created: {source_log}')

        saved_log = result_dir / f'run_{run_index}.log'
        shutil.copy2(source_log, saved_log)
        log_text = saved_log.read_text(encoding='utf-8')
        console_text = console_log.read_text(encoding='utf-8')
        try:
            main_values = find_accuracy_array(log_text, 'Main', expected_rounds)
            attack_values = find_accuracy_array(log_text, 'Attacker', expected_rounds)
        except RuntimeError:
            # Fall back to captured console output if the file logger format differs.
            main_values = find_accuracy_array(console_text, 'Main', expected_rounds)
            attack_values = find_accuracy_array(console_text, 'Attacker', expected_rounds)

        main_accuracies.append(main_values)
        attack_accuracies.append(attack_values)
        print(f'[{run_index}/{args.runs}] Saved {saved_log.name}', flush=True)

    write_per_run_csv(
        result_dir / 'per_run_accuracy.csv',
        main_accuracies,
        attack_accuracies,
    )
    write_mean_csv(
        result_dir / 'mean_accuracy.csv',
        main_accuracies,
        attack_accuracies,
    )

    print(f'Finished. Results: {result_dir}', flush=True)


if __name__ == '__main__':
    main()
