"""Download and verify the pinned LocalgovFAQ derivative into an ignored folder."""
import argparse
import hashlib
import pathlib
import urllib.request

from independent_eval import BASE, BLOBS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=pathlib.Path,
                        default=pathlib.Path('work/localgovfaq/dataset'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in BLOBS.items():
        target = args.output_dir / name
        content = (target.read_bytes() if target.exists()
                   else urllib.request.urlopen(f'{BASE}/{name}', timeout=60).read())
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: unexpected source hash {actual}; no file written')
        if not target.exists():
            target.write_bytes(content)
        print(f'Verified {name}: {len(content)} bytes')


if __name__ == '__main__':
    main()
