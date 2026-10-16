"""Apply frozen training parameters to preprocessed scans, without re-fitting."""
import argparse
from pathlib import Path
from common import write_csv
from pet_experiment import read_images
from ssm import SSM
import numpy as np

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--images', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    X, _, _ = read_images(args.images, str(args.model_dir/'mask.nii.gz'))
    with np.load(args.model_dir/'model.npz', allow_pickle=False) as m:
        pcs = (SSM.subject_center(X) - m['group_mean'] - m['pca_mean']) @ m['components'].T
        expression = pcs @ m['coefficient']
        decision = expression + m['intercept'][0]
        z = (expression - m['hc_mean']) / m['hc_std']
    write_csv(args.out, [{'image': path, 'expression': float(e), 'decision': float(d), 'z_expression': float(s)}
                        for path,e,d,s in zip(args.images,expression,decision,z)])
