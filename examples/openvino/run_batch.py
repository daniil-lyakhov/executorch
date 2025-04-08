import sys

import pandas as pd
from aot_optimize_and_infer import main as aot_main
from executorch.backends.openvino.quantizer.quantizer import QuantizationMode

import nncf

MODELS = (
(
        "torchvision",
        ("regnet_x_32gf", {}, {}),
    ),
    (
        "torchvision",
        ("mnasnet1_3", {}, {}),
    ),
    (
        "torchvision",
        ("resnet50", {}, {}),
    ),
    (
        "timm",
        ("dpn68", {}, {}),
    ),
)


def main(dataset_path: str):
    result = []
    for suite, (model_name, quantizer_kwargs, quantize_pt2e_kwargs) in MODELS:
        for quantize in [True]:
            try:
                print(30 * "*")
                print(f"START {suite} {model_name} {quantizer_kwargs} {quantize_pt2e_kwargs} q:{quantize}")
                print(30 * "*")
                with nncf.torch.disable_patching():
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
