import sys

import pandas as pd
from aot_xnnpack_compiler import main as aot_main


MODELS = (
    (
        "torchvision",
        ("mobilenet_v2", {}, {"fast_bias_correction": False}),
    ),
    (
        "torchvision",
        ("mobilenet_v3_small", {}, {"fast_bias_correction": False}),
    ),
    (
        "torchvision",
        ("resnet18", {}, {}),
    ),
    (
        "torchvision",
        ("resnet50", {}, {}),
    ),
    (
        "torchvision",
        ("vit_b_16", {}, {"smooth_quant": True}),
    ),
)


def main(dataset_path: str, path_to_runner: str):
    result = []
    for suite, (model_name, _, _) in MODELS:
        for quantize in [False, True]:

            try:
                print(30 * "*")
                print(f"START {suite} {model_name}  q:{quantize}")
                print(30 * "*")
                res = aot_main(
                    suite=suite,
                    model_name=model_name,
                    input_shape=None,
                    quantize=quantize,
                    validate=True,
                    dataset_path=dataset_path,
                    path_to_runner=path_to_runner,
                    batch_size=125,
                    device="CPU",
                )
                print(30 * "*")
                print(f"{suite} {model_name} q:{quantize} -> {res}")
                print(30 * "*")
            except Exception as e:
                print(30 * "*")
                print(f"ERROR: {suite} {model_name} q:{quantize} {e}")
                print(30 * "*")
                res = e

            res_t = (suite, model_name, quantize, res)
            result.append(res_t)

    df = pd.DataFrame(
        result,
        columns=["suite", "model_name", "quantize", "acc"],
    )

    print(df)
    df.to_csv("result.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
