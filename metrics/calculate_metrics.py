import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from metrics.metric_utils import (  # noqa: E402
    build_lpips_model,
    build_no_reference_models,
    build_unique_name_index,
    calculate_lpips_from_paths,
    calculate_no_reference_from_path,
    iter_output_images,
    mean_metric,
    read_image_rgb,
    resolve_gt_path,
    write_csv,
)
from utils import utils_image as util  # noqa: E402


FULL_REFERENCE_METRICS = {'psnr', 'ssim', 'lpips'}
NO_REFERENCE_METRICS = {'niqe', 'nrqm', 'pi'}
ALL_METRICS = FULL_REFERENCE_METRICS | NO_REFERENCE_METRICS


def parse_args():
    parser = argparse.ArgumentParser(description='Compute SelfDRSC++ evaluation metrics from saved images.')
    parser.add_argument('--output', required=True, help='Root folder of generated results.')
    parser.add_argument('--gt', default=None, help='Root folder of GT videos/images. Required for PSNR/SSIM/LPIPS.')
    parser.add_argument('--gt-subdir', default='GS', help='GT subfolder under each RS-GOPRO video folder.')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='Metrics to compute. Default: psnr ssim lpips when --gt is set; otherwise niqe nrqm pi.')
    parser.add_argument('--crop-border', type=int, default=20, help='Crop border for PSNR/SSIM.')
    parser.add_argument('--lpips-crop-border', type=int, default=0, help='Crop border for LPIPS.')
    parser.add_argument('--nr-crop-border', type=int, default=20, help='Crop border for NIQE/NRQM/PI.')
    parser.add_argument('--lpips-net', default='alex', choices=['alex', 'vgg', 'squeeze'], help='LPIPS backbone.')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device for LPIPS and pyiqa metrics.')
    parser.add_argument('--save-csv', default=None, help='Optional per-image CSV path.')
    parser.add_argument('--save-summary', default=None, help='Optional per-video summary CSV path.')
    return parser.parse_args()


def validate_metrics(args):
    if args.metrics is None:
        metrics = ['psnr', 'ssim', 'lpips'] if args.gt else ['niqe', 'nrqm', 'pi']
    else:
        metrics = [metric.lower() for metric in args.metrics]

    unknown = sorted(set(metrics) - ALL_METRICS)
    if unknown:
        raise ValueError('Unknown metric(s): {}. Supported metrics are: {}.'.format(
            ', '.join(unknown), ', '.join(sorted(ALL_METRICS))))

    if (set(metrics) & FULL_REFERENCE_METRICS) and args.gt is None:
        raise ValueError('Full-reference metrics ({}) require --gt.'.format(
            ', '.join(sorted(set(metrics) & FULL_REFERENCE_METRICS))))
    return metrics


def build_metric_models(metrics, args):
    lpips_model, lpips_device = None, None
    nr_models = None

    if 'lpips' in metrics:
        lpips_model, lpips_device = build_lpips_model(net=args.lpips_net, device=args.device)

    nr_metric_names = [metric for metric in metrics if metric in NO_REFERENCE_METRICS]
    if nr_metric_names:
        nr_models = build_no_reference_models(
            nr_metric_names, device=args.device, crop_border_value=args.nr_crop_border)

    return lpips_model, lpips_device, nr_models


def calculate_rows(args, metrics):
    output_root = Path(args.output)
    gt_root = Path(args.gt) if args.gt else None
    lpips_model, lpips_device, nr_models = build_metric_models(metrics, args)
    gt_index = build_unique_name_index(gt_root) if gt_root else None
    rows = []

    for video_name, output_path in iter_output_images(output_root):
        row = {
            'video': video_name,
            'image': output_path.name,
            'output': str(output_path),
        }
        gt_path = None
        if set(metrics) & FULL_REFERENCE_METRICS:
            gt_path = resolve_gt_path(output_root, output_path, gt_root, gt_subdir=args.gt_subdir, gt_index=gt_index)
            row['gt'] = str(gt_path)
            output_img = read_image_rgb(output_path)
            gt_img = read_image_rgb(gt_path)
            if 'psnr' in metrics:
                row['psnr'] = util.calculate_psnr(output_img, gt_img, border=args.crop_border)
            if 'ssim' in metrics:
                row['ssim'] = util.calculate_ssim(output_img, gt_img, border=args.crop_border)
            if 'lpips' in metrics:
                row['lpips'] = calculate_lpips_from_paths(
                    output_path, gt_path, lpips_model, lpips_device, crop=args.lpips_crop_border)
        else:
            row['gt'] = ''

        if nr_models is not None:
            row.update(calculate_no_reference_from_path(output_path, nr_models))
        rows.append(row)

    if not rows:
        raise RuntimeError('No output images found under {}.'.format(output_root))
    return rows


def summarize(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['video']].append(row)

    summary_rows = []
    for video_name in sorted(grouped):
        video_rows = grouped[video_name]
        row = {'video': video_name, 'num_images': len(video_rows)}
        for metric in metrics:
            row[metric] = mean_metric(video_rows, metric)
        summary_rows.append(row)

    avg_row = {'video': 'AVG', 'num_images': len(rows)}
    for metric in metrics:
        avg_row[metric] = mean_metric(rows, metric)
    summary_rows.append(avg_row)
    return summary_rows


def print_summary(summary_rows, metrics):
    for row in summary_rows:
        metric_text = []
        for metric in metrics:
            value = row.get(metric)
            if value is not None and not np.isnan(value):
                metric_text.append('{}={:.5f}'.format(metric.upper(), value))
        print('{} ({} images): {}'.format(row['video'], row['num_images'], ', '.join(metric_text)))


def main():
    args = parse_args()
    metrics = validate_metrics(args)
    rows = calculate_rows(args, metrics)
    summary_rows = summarize(rows, metrics)

    fieldnames = ['video', 'image', 'output', 'gt'] + metrics
    summary_fieldnames = ['video', 'num_images'] + metrics
    if args.save_csv:
        write_csv(args.save_csv, rows, fieldnames)
    if args.save_summary:
        write_csv(args.save_summary, summary_rows, summary_fieldnames)
    elif args.save_csv:
        csv_path = Path(args.save_csv)
        write_csv(csv_path.with_name(csv_path.stem + '_summary' + csv_path.suffix), summary_rows, summary_fieldnames)

    print_summary(summary_rows, metrics)


if __name__ == '__main__':
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    main()
