"""Run a synthetic SSM/PCA experiment, or grouped CV on preprocessed real FDG PET.

Real input CSV: subject_id,label,path (0=HC, 1=PD); one row per scan.
Images and an independently defined mask MUST already share anatomical space.
"""
import argparse
import csv
from pathlib import Path
from common import ROOT, save_json, write_csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, gaussian_filter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, roc_curve
from ssm import PatternModel


def read_images(paths, mask_path):
    mask_img = nib.load(mask_path)
    mask = np.asarray(mask_img.dataobj) > 0
    if mask.ndim != 3 or mask.sum() < 10:
        raise ValueError('Need a nonempty 3D brain mask')
    matrix = []
    for path in paths:
        img = nib.load(path)
        if img.shape != mask.shape or not np.allclose(img.affine, mask_img.affine, atol=1e-4):
            raise ValueError(f'Grid mismatch: {path}; register to the mask space first')
        values = img.get_fdata(dtype=np.float32)[mask]
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f'Nonpositive/nonfinite uptake inside mask: {path}')
        matrix.append(values)
    return np.asarray(matrix), mask, mask_img.affine


def synthetic():
    """Real normal template as background; all 100 subject images/labels are simulated."""
    rng = np.random.default_rng(42)
    img = nib.load(ROOT / 'data/cnpet/CN_PET_TEMPLATE.nii')
    base = img.get_fdata()
    shape = (24, 28, 24)
    base = zoom(base, np.array(shape) / base.shape, order=1)
    mask = base > base.max() * .18
    coords = np.indices(shape).astype(float)
    coords = [(coords[i] - (shape[i] - 1) / 2) / (shape[i] / 2) for i in range(3)]
    a, b, c = coords
    # Toy blobs are deliberately NOT labelled as anatomical nuclei or a published PDRP.
    pattern = np.exp(-((a-.3)**2+(b+.2)**2+(c+.2)**2)/.09)
    pattern += np.exp(-((a+.3)**2+(b+.2)**2+(c+.2)**2)/.09)
    pattern -= np.exp(-(a*a+(b-.4)**2+(c-.25)**2)/.22)
    pattern = pattern[mask]
    pattern -= pattern.mean()
    pattern /= np.linalg.norm(pattern)
    nuisance = np.stack([a[mask], b[mask], c[mask]])
    nuisance /= np.linalg.norm(nuisance, axis=1, keepdims=True)
    y = np.repeat([0, 1], 50)
    rows = []
    for label in y:
        noise = gaussian_filter(rng.normal(size=shape), .7)[mask]
        log_image = np.log(base[mask]) + rng.normal(0, .25)
        log_image += rng.normal(0, .4, 3) @ nuisance
        log_image += rng.normal(label * 1.5, .65) * pattern + .04 * noise
        rows.append(np.exp(log_image))
    # Downsampled template retains physical field of view, including voxel centre shift.
    transform = np.eye(4)
    scale = np.asarray(img.shape) / shape
    transform[:3, :3] = np.diag(scale)
    transform[:3, 3] = (scale - 1) / 2
    return np.asarray(rows), y, np.array([f'SIM-{i:03}' for i in range(len(y))]), mask, img.affine @ transform, pattern


def evaluate(X, y, groups, out, mask, affine, kind, truth=None):
    out.mkdir(parents=True, exist_ok=True)
    unique = np.unique(groups)
    group_labels = []
    for group in unique:
        labels = np.unique(y[groups == group])
        if len(labels) != 1:
            raise ValueError(f'Conflicting diagnosis for subject {group}')
        group_labels.append(labels[0])
    if min(np.bincount(group_labels, minlength=2)) < 5:
        raise ValueError('Five-fold CV needs at least five independent subjects per class')
    scores, zscores = np.zeros(len(y)), np.zeros(len(y))
    folds = np.zeros(len(y), dtype=int)
    fold_stats = []
    for fold, (train, test) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups), 1):
        assert not set(groups[train]) & set(groups[test])
        if set(y[test]) != {0, 1}:
            raise ValueError('One-class test fold: adjust cohort/split design')
        model = PatternModel(3).fit(X[train], y[train])
        scores[test], zscores[test], folds[test] = model.score(X[test]), model.zscore(X[test]), fold
        fold_stats.append({'fold': fold, 'test_subjects': len(set(groups[test])),
                           'auc': float(roc_auc_score(y[test], scores[test]))})
    # Report primary metrics at SUBJECT level, averaging repeated-scan decision values.
    subject_rows = []
    for g in unique:
        at = groups == g
        subject_rows.append({'subject_id': g, 'label': int(y[at][0]), 'fold': int(folds[at][0]),
                             'decision': float(scores[at].mean()), 'z_expression': float(zscores[at].mean())})
    sy = np.array([r['label'] for r in subject_rows])
    sp = np.array([r['decision'] for r in subject_rows])
    tn, fp, fn, tp = confusion_matrix(sy, sp >= 0, labels=[0, 1]).ravel()
    metrics = {'data_kind': kind, 'n_scans': len(y), 'n_subjects': len(unique),
               'n_features': X.shape[1], 'fixed_n_components': 3,
               'subject_oof_auc': float(roc_auc_score(sy, sp)),
               'subject_balanced_accuracy': float(balanced_accuracy_score(sy, sp >= 0)),
               'sensitivity': float(tp/(tp+fn)), 'specificity': float(tn/(tn+fp)),
               'confusion_matrix_HC_PD': [[int(tn), int(fp)], [int(fn), int(tp)]], 'folds': fold_stats,
               'note': 'Each fold fits SSM/PCA, logistic weights and HC reference on training data only. OOF z scores have fold-specific references. Final all-data map is for future validation.'}
    final = PatternModel(3).fit(X, y)
    if truth is not None:
        metrics['simulated_pattern_correlation_final_training_fit'] = float(np.corrcoef(final.pattern_, truth)[0, 1])
    save_json(out / 'metrics.json', metrics)
    write_csv(out / 'subject_oof_predictions.csv', subject_rows)
    write_csv(out / 'scan_oof_predictions.csv', [{'subject_id': g, 'label': int(t), 'fold': int(f), 'decision': float(s), 'z_expression': float(z)}
              for g,t,f,s,z in zip(groups,y,folds,scores,zscores)])
    volume = np.zeros(mask.shape, dtype=np.float32)
    volume[mask] = final.pattern_
    nib.save(nib.Nifti1Image(volume, affine), out / 'training_pattern.nii.gz')
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), out / 'mask.nii.gz')
    # Portable parameters, no pickle / untrusted model deserialization.
    np.savez_compressed(out / 'model.npz', group_mean=final.ssm.group_mean_,
        pca_mean=final.ssm.pca_.mean_, components=final.ssm.pca_.components_,
        coefficient=final.classifier.coef_[0], intercept=final.classifier.intercept_,
        hc_mean=final.hc_mean_, hc_std=final.hc_std_)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    fpr, tpr, _ = roc_curve(sy, sp)
    axes[0].plot(fpr, tpr, label=f'OOF AUC={metrics["subject_oof_auc"]:.3f}')
    axes[0].plot([0,1],[0,1], '--', color='gray'); axes[0].legend()
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
    axes[1].boxplot([[r['z_expression'] for r in subject_rows if r['label']==c] for c in [0,1]], tick_labels=['HC','PD'])
    axes[1].set(ylabel='Out-of-fold expression z score')
    vmax = np.max(np.abs(volume))
    axes[2].imshow(np.rot90(volume[:,:,volume.shape[2]//2]), cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[2].set_title('All-data training pattern'); axes[2].axis('off')
    fig.suptitle(kind); fig.tight_layout();fig.savefig(out / 'experiment.png', dpi=160);plt.close(fig)
    print(metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path)
    p.add_argument('--mask', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.manifest:
        if not args.mask:
            p.error('--mask is required for real data')
        with args.manifest.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        paths = [str((args.manifest.parent / r['path']).resolve()) for r in rows]
        X, mask, affine = read_images(paths, str(args.mask))
        y = np.array([int(r['label']) for r in rows])
        if set(y) != {0,1}:
            raise ValueError('labels must contain only HC=0 and PD=1')
        groups = np.array([r['subject_id'] for r in rows])
        evaluate(X, y, groups, args.out or ROOT/'outputs/real_pet', mask, affine, 'REAL_FDG_PET_EXPLORATORY')
    else:
        X, y, groups, mask, affine, truth = synthetic()
        evaluate(X, y, groups, args.out or ROOT/'outputs/synthetic_pet', mask, affine,
                 'SYNTHETIC SUBJECTS - NOT CLINICAL PD RESULTS', truth)
