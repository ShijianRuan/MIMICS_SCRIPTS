"""
probe_onnx.py — 探测 RONDON 的 DINOv3 ViT-S/16 ONNX 编码器的输入输出。

来源/用途：
  RONDON 的编码器以 ONNX 形式存储于
  C:\\Users\\shijian.ruan\\AppData\\Roaming\\RONDON\\model\\dinov3-vits16\\model.onnx
  本脚本用 onnxruntime 实跑，确认 patch 策略的关键事实：
    - 输入形状 (B, 3, 512, 512)  -> 3 通道 RGB（医学影像单通道需先复制成 3 通道）
    - 输出形状 (B, 384, 32, 32)  -> patch_size=16, grid=32x32, embed_dim=384 (ViT-S)
    - 3 个输出 hidden1/hidden2/output，训练取 output

用法（用 RONDON 自带的嵌入式 Python）：
  "C:\\Users\\shijian.ruan\\AppData\\Local\\RONDON\\python.exe" probe_onnx.py

证据：本脚本的输出已记录在 docs/01_REVERSE_ENGINEERING.md 第 3.1 节。
"""
import sys
import numpy as np
import onnxruntime as ort

ONNX_PATH = r"C:\Users\shijian.ruan\AppData\Roaming\RONDON\model\dinov3-vits16\model.onnx"


def main():
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    print("=== INPUTS ===")
    for i in sess.get_inputs():
        print(f"  {i.name}  shape={i.shape}  type={i.type}")

    print("=== OUTPUTS ===")
    for o in sess.get_outputs():
        print(f"  {o.name}  shape={o.shape}  type={o.type}")

    print("=== providers ===", sess.get_providers())

    # 实跑一个零输入，探测真实输出形状与值域
    x = np.zeros((1, 3, 512, 512), dtype=np.float32)
    outs = sess.run(None, {sess.get_inputs()[0].name: x})
    print("=== RUN (zero input (1,3,512,512)) ===")
    for o, v in zip(sess.get_outputs(), outs):
        print(f"  {o.name} -> {v.shape}  range[{v.min():.3f}, {v.max():.3f}]")

    # 验证灰度->3通道复制
    gray = np.zeros((512, 512), dtype=np.float32)
    rgb = np.stack([gray, gray, gray], axis=0)[None]  # (1,3,512,512)
    assert rgb.shape == (1, 3, 512, 512), rgb.shape
    print("=== gray->rgb reshape OK:", rgb.shape, "===")


if __name__ == "__main__":
    sys.exit(main())
