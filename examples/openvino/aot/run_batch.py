import sys

import pandas as pd
from aot_openvino_compiler import main as aot_main
from executorch.backends.openvino.quantizer.quantizer import QuantizationMode
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters

import nncf

MODELS = (
    (
        "torchvision",
        ("mobilenet_v2", {"mode": QuantizationMode.INT8_MIXED}, {"fast_bias_correction": False}),
    ),
    (
        "torchvision",
        ("mobilenet_v3_small", {"mode": QuantizationMode.INT8_MIXED}, {"fast_bias_correction": False}),
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
        ("vit_b_16", {"mode": QuantizationMode.INT8_TRANSFORMER}, {"smooth_quant": True, "smooth_quant_params": AdvancedSmoothQuantParameters(matmul=0.15)}),
    ),
)[-1:]


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
                        quantization_flow="nncf",
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
