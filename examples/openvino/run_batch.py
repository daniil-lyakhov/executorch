import sys

import pandas as pd
from aot_optimize_and_infer import main as aot_main


MODELS = (
(
        "torchvision",
        ("vit_b_16", {}, {}),
    ),
    (
        "torchvision",
        ("swin_v2_s", {}, {}),
    ),
    (
        "torchvision",
        ("resnet50", {}, {}),
    ),
    (
        "torchvision",
        ("mobilenet_v3_small", {}, {}),
    ),
    (
        "timm",
        ("deit3_small_patch16_224_in21ft1k", {}, {})
    ),
    (
        "timm",
        ("ese_vovnet39b", {}, {})
    )
)


def main(dataset_path: str):
    result = []
    for suite, (model_name, quantizer_kwargs, quantize_pt2e_kwargs) in MODELS:
        for quantize in [True]:
            try:
                print(30 * "*")
                print(f"START {suite} {model_name} {quantizer_kwargs} {quantize_pt2e_kwargs} q:{quantize}")
                print(30 * "*")
                res = aot_main(
                    suite=suite,
                    model_name=model_name,
                    input_shape=None,
                    save_model=False,
                    model_file_name="",
                    quantize=quantize,
                    validate=True,
                    dataset_path=dataset_path,
                    batch_size=125,
                    device="CPU",
                    infer=False,
                    num_iter=1,
                    warmup_iter=1,
                    input_path="",
                    output_path="",
                )
                print(30 * "*")
                print(f"{suite} {model_name} {quantizer_kwargs} {quantize_pt2e_kwargs} q:{quantize} -> {res}")
                print(30 * "*")
            except Exception as e:
                print(30 * "*")
                print(f"ERROR: {suite} {model_name} {quantizer_kwargs} {quantize_pt2e_kwargs} q:{quantize} {e}")
                print(30 * "*")
                res = e

            res_t = (suite, model_name, quantizer_kwargs, quantize_pt2e_kwargs, quantize, res)
            result.append(res_t)

    df = pd.DataFrame(
        result,
        columns=["suite", "model_name", "quantizer_kwargs", "quantize_pt2e_kwargs", "quantize", "acc"],
    )

    print(df)
    df.to_csv("result.csv")


if __name__ == "__main__":
    main(sys.argv[1])
