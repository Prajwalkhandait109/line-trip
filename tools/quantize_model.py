"""
Run this on Windows to produce an ONNX-optimized model compatible with cv2.dnn.
onnxoptimizer folds constants and removes redundant ops — typically 10-20% faster.

Usage:
    pip install onnxoptimizer
    python tools/quantize_model.py
"""
import os
import onnx
import onnxoptimizer

MODELS = [
    ("models/yolov5n_320.onnx", "models/yolov5n_320_opt.onnx"),
]

# Safe passes: constant folding + redundant op elimination, no quantization ops
PASSES = [
    "eliminate_deadend",
    "eliminate_identity",
    "eliminate_nop_dropout",
    "eliminate_nop_flatten",
    "eliminate_nop_monotone_argmax",
    "eliminate_nop_pad",
    "eliminate_nop_transpose",
    "eliminate_unused_initializer",
    "fuse_add_bias_into_conv",
    "fuse_bn_into_conv",
    "fuse_consecutive_concats",
    "fuse_consecutive_log_softmax",
    "fuse_consecutive_reduce_unsqueeze",
    "fuse_consecutive_squeezes",
    "fuse_consecutive_transposes",
    "fuse_matmul_add_bias_into_gemm",
    "fuse_pad_into_conv",
    "fuse_transpose_into_gemm",
]

for src, dst in MODELS:
    if not os.path.exists(src):
        print("SKIP (not found): {}".format(src))
        continue
    print("Optimizing {} -> {} ...".format(src, dst))
    model = onnx.load(src)
    try:
        optimized = onnxoptimizer.optimize(model, PASSES)
    except Exception as e:
        print("  Some passes failed ({}), retrying with defaults...".format(e))
        optimized = onnxoptimizer.optimize(model)
    onnx.save(optimized, dst)
    src_mb = os.path.getsize(src) / 1e6
    dst_mb = os.path.getsize(dst) / 1e6
    print("  Done. {:.1f} MB -> {:.1f} MB".format(src_mb, dst_mb))
    print("  Transfer to Pi:")
    print("    scp {} pi@10.129.2.70:~/line_cross/line-trip/{}".format(dst, dst))
