import sys

import pandas as pd
from aot_openvino_compiler import main as aot_main

import nncf

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


def main(dataset_path: str):
    result = []
    for suite, (model_name, quantizer_kwargs, quantize_pt2e_kwargs) in MODELS:
        for quantize in [True, False]:

            try:
                print(30 * "*")
                print(f"START {suite} {model_name} {quantizer_kwargs} {quantize_pt2e_kwargs} q:{quantize}")
                print(30 * "*")
                with nncf.torch.disable_patching():
                    res = aot_main(
                        suite=suite,
                        model_name=model_name,
                        input_shape=None,
                        quantize=quantize,
                        validate=True,
                        dataset_path=dataset_path,
                        batch_size=125,
                        device="CPU",
                        quantizer_kwargs=quantizer_kwargs,
                        quantize_pt2e_kwargs=quantize_pt2e_kwargs,
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
