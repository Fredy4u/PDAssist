"""Download publisher-hosted data; record provenance and SHA256 of actual bytes."""
import argparse
import hashlib
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, save_json

SOURCES = {
    'pet': [
        ('cnpet/Normalized_images.zip', 'https://www.nitrc.org/frs/download.php/12231/Normalized_images.zip?i_agree=1&release_id_4471=1'),
        ('cnpet/Subjects_Information.csv', 'https://www.nitrc.org/frs/download.php/12117/Subjects_Information.csv?i_agree=1&release_id_4471=1'),
        ('cnpet/source_README.md', 'https://www.nitrc.org/frs/download.php/12232/README.md?i_agree=1&release_id_4471=1'),
        ('cnpet/LICENSE', 'https://www.nitrc.org/docman/view.php/1486/173511/LICENSE'),
        *[(f'cnpet/{name}', f'https://raw.githubusercontent.com/DlutMedimgGroup/Chinese-Brain-PET-Template/master/Template/{name}')
          for name in ['CN_PET_TEMPLATE.nii', 'CN_PET_TEMPLATE_MASK.nii', 'CN_PET_TEMPLATE_STDV.nii']],
    ],
}


def fetch(relative, url):
    path = ROOT / 'data' / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.part')
        for attempt in range(3):
            try:
                download_url = url.replace('release_id_4471=1', 'download_now=1')
                request = urllib.request.Request(download_url, headers={'User-Agent': 'PD-research-starter/1.0'})
                with urllib.request.urlopen(request, timeout=90) as response, temp.open('wb') as out:
                    length = response.headers.get('Content-Length')
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > 350 * 1024**2:
                            raise ValueError('Download exceeds per-file limit of 350 MiB')
                        out.write(chunk)
                    if length and int(length) != size:
                        raise ValueError('Incomplete download')
                if path.suffix == '.zip' and not zipfile.is_zipfile(temp):
                    raise ValueError('Expected ZIP; server may have returned a login page')
                if temp.read_bytes()[:100].lstrip().startswith((b'<?xml', b'<!DOCTYPE', b'<html')):
                    raise ValueError('Server returned an HTML page rather than data')
                temp.replace(path)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    if path.suffix == '.zip':
        destination = path.with_suffix('')
        with zipfile.ZipFile(path) as zf:
            if sum(info.file_size for info in zf.infolist()) > 2 * 1024**3:
                raise ValueError('Archive exceeds 2 GiB extraction limit')
            for info in zf.infolist():
                target = (destination / info.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Unsafe archive path')
            zf.extractall(destination)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    print(f'{relative}: {path.stat().st_size:,} bytes', flush=True)
    return {'path': relative, 'url': url, 'bytes': path.stat().st_size, 'sha256': digest,
            'verified_at_utc': datetime.now(timezone.utc).isoformat()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['pet'], default='pet',
                        help='download the Chinese normal FDG-PET template and cohort for pipeline development')
    args = parser.parse_args()
    manifest = ROOT / 'data/download_manifest.json'
    import json
    records = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
    for relative, url in SOURCES[args.dataset]:
        record = fetch(relative, url)
        records = [r for r in records if r['path'] != relative] + [record]
        save_json(manifest, records)
