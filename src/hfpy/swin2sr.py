import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
from transformers.utils import cached_file

model_id = "thanhhau097/swin2SR-realworld-sr-x4-64-bsrgan-psnr"


def run(image_path: str):
    processor = AutoImageProcessor.from_pretrained(model_id)

    # 1. 重みファイルを直接キャッシュから特定して読み込む
    # (transformers v5のAutoModel自動変換を迂回するため、Swin2SRForImageSuperResolutionを明示的に使用)
    try:
        # まずは通常通りロードを試みる
        model = Swin2SRForImageSuperResolution.from_pretrained(model_id)
    except Exception:
        # 万が一ロード自体で弾かれる場合は空のモデルを作って手動ロードします
        from transformers import Swin2SRConfig

        config = Swin2SRConfig.from_pretrained(model_id)
        model = Swin2SRForImageSuperResolution(config)

    # 2. Hugging Faceのキャッシュから本来の pytorch_model.bin を探し出す
    archive_file = cached_file(model_id, "pytorch_model.bin")
    state_dict = torch.load(archive_file, map_location="cpu")

    # 3. 古いキー名をプログラムが要求する新しいキー名にマッピング（書き換え）
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # もし古いキー名(conv_first)があれば新しいキー名(first_convolution)に変換
        if "swin2sr.conv_first" in key:
            new_key = key.replace("swin2sr.conv_first", "swin2sr.first_convolution")
        new_state_dict[new_key] = value

    # 4. 正しい名前になった重みを厳格に（strict=Trueで）モデルに適用
    model.load_state_dict(new_state_dict, strict=True)
    print("Swin2SRのキー名を正常に修正し、完全にロードしました。")

    # 5. 画像処理と推論の実行
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # reconstruction からデータを抽出
    output_tensor = outputs.reconstruction.data.squeeze().cpu().clamp(0, 1)
    output_tensor = output_tensor.moveaxis(0, -1).numpy() * 255.0
    output_image = Image.fromarray(output_tensor.astype("uint8"))

    save_path = "output_scaled.png"
    output_image.save(save_path)
    print(f"保存しました。save_path={save_path}")
