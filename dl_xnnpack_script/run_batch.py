import sys

import pandas as pd
from aot_xnnpack_compiler import main as aot_main

MODELS = (
    (
        "torchvision",
        ("efficientnet_b0", {}, {}),
    ),
    (
        "torchvision",
        ("inception_v3", {}, {}),
    ),
(
        "torchvision",
        ("mnasnet1_3", {}, {}),
    ),
(
        "torchvision",
        ("regnet_x_32gf", {}, {}),
    ),
(
        "torchvision",
        ("wide_resnet101_2", {}, {}),
    ),
(
        "torchvision",
        ("wide_resnet50_2", {}, {}),
    ),
    (
        "torchvision",
        ("resnet50", {}, {}),
    ),
(
        "timm",
        ("fbnetc_100.rmsp_in1k", {}, {}),
    ),
(
        "timm",
        ("densenetblur121d", {}, {}),
    ),(
        "timm",
        ("res2net50_26w_4s", {}, {}),
    ),
    (
        "timm",
        ("regnetz_b16", {}, {}),
    ),
    (
        "timm",
        ("dpn68", {}, {}),
    ),
(
        "timm",
        ("mixnet_l.ft_in1k", {}, {}),
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
                    validate=False,
                    dataset_path=dataset_path,
                    path_to_runner=path_to_runner,
                    batch_size=1,
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
    df.to_csv("result_xnnpack.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
