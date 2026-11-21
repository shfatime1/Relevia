"""Stage only the existing artifacts needed by the Hugging Face Space."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
TARGET = ROOT / '.cache' / 'relevia-space'


def copy_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    for name in ('app.py', 'README.md', 'requirements.txt'):
        copy_file(SOURCE / name, TARGET / name)
    for name in ('api.py', 'image_search.py', 'recommend.py', 'search.py', 'sentiment.py', 'unified_search.py'):
        copy_file(ROOT / 'backend' / name, TARGET / 'backend' / name)
    for name in ('products.parquet', 'reviews.parquet', 'review_sentiment.parquet', 'product_sentiment.parquet'):
        copy_file(ROOT / 'data' / name, TARGET / 'data' / name)
    images = list((ROOT / 'data' / 'images').glob('*.webp'))
    if len(images) != 1120:
        raise ValueError(f'Expected 1,120 WebP images, found {len(images)}')
    for image in images:
        copy_file(image, TARGET / 'data' / 'images' / image.name)
    for kind in ('text_search', 'image_search'):
        for name in ('manifest.json', 'embeddings.npy', 'products.faiss'):
            copy_file(ROOT / '.cache' / kind / name, TARGET / '.cache' / kind / name)
    if not (ROOT / 'frontend' / 'dist' / 'index.html').is_file():
        raise FileNotFoundError('Build the production frontend before staging')
    shutil.copytree(ROOT / 'frontend' / 'dist', TARGET / 'dist', dirs_exist_ok=True)
    print(f'Staged {len(images)} images and existing artifacts in {TARGET}')


if __name__ == '__main__':
    main()
