import argparse
from . import swin2sr


def main() -> None:
    parser = argparse.ArgumentParser(description="hfpyを実行する")
    parser.add_argument("image_path", type=str, help="画像のパス")
    args = parser.parse_args()

    image_path = args.image_path
    swin2sr.run(image_path)
