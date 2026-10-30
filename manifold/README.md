# atlas_eval：用 Atlas 带位姿 RGBD 视图评测 InfiniSplat 的 splat

Manifold 在 InfiniSplat fork 里加的评测脚本，**不改上游文件**。输入是「场景文件夹 + `manifest.json`」：

- **单图**（1 个输入视图）走 v1，可以只用 RGB，也可以把深度当 depth prompt 一起给；
- **多图**（2–50 个输入视图）走 v2，可以用给定相机、DA3 估位姿，或完全不用位姿；
- **全景**用 v2 自带的 `infinisplat-panorama`（[docs/panorama.md](../InfiniSplat_v2/docs/panorama.md)），这里不另写脚本。

渲染、打分和汇总对三种来源一视同仁，外部做出来的 splat（比如面元基线）也能登记进来一起比。

## 场景文件夹

Atlas scene 导出原样放进一个文件夹，文件名沿用导出时的名字：

```text
manifold/data/S1_living_room/
├── atlas-scene.json                    # Atlas scene 元数据（schema_version 1）
├── scene-v_0123456789ab.png            # 视图 RGB，1280×720
├── scene-v_0123456789ab-depth.exr      # 线性 z 深度（EXR，NaN 表示无几何），或 log 深度 PNG
├── scene-v_0123456789ab-confidence.png # 置信度（可能没有）
├── ...
└── manifest.json                       # 由 make_manifest 生成，也可以手写
```

普通照片（不带相机）也行：照片直接放进文件夹，跑 `make_manifest images`。`manifold/data/` 和 `manifold/runs/` 已加入 `.gitignore`，场景数据和输出不会进仓。

## manifest.json

`schema` 固定为 `infinisplat-atlas-eval/1`。路径一律相对场景文件夹，不能是绝对路径，也不能含 `..`。不认识的键会被忽略。

```json
{
  "schema": "infinisplat-atlas-eval/1",
  "name": "S1_living_room",
  "source": {"kind": "atlas_scene", "file": "atlas-scene.json", "sha256": "…", "coordinate_system": "rub",
             "split": {"test_create": 1, "test_every": 4}},
  "views": [
    {
      "view_id": "v_0123456789ab",
      "origin": "create",
      "role": "input",
      "image": "scene-v_0123456789ab.png",
      "camera": {
        "intrinsics": {"fx": 912.4, "fy": 912.4, "cx": 640.0, "cy": 360.0, "width": 1280, "height": 720},
        "extrinsics": {"position": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                       "coordinate_system": "rub"}
      },
      "depth": {"file": "scene-v_0123456789ab-depth.exr", "encoding": "linear", "format": "exr"},
      "confidence": "scene-v_0123456789ab-confidence.png"
    }
  ]
}
```

每个视图的字段：

| 字段 | 必填 | 取值 | 说明 |
|---|---|---|---|
| `view_id` | 是 | `[A-Za-z0-9][A-Za-z0-9_.-]*`，不能重复 | Atlas 视图 id 原样照抄 |
| `role` | 是 | `input` / `test` | `input` 交给重建，`test` 只用来打分 |
| `origin` | 否 | `create` / `generate` / `photo`（缺省） | `create` 是 Atlas 由真实照片 reframe 出来的帧；`generate` 是 Atlas 生成的新视角；打分时按它分组 |
| `image` | 是 | PNG / JPG / WebP | 有 `camera.intrinsics.width/height` 时，图像尺寸必须与之相同 |
| `camera` | 否 | 与 Atlas 的 camera 同形 | `intrinsics` 是像素单位的 `fx fy cx cy`（可加 `width height`）；`extrinsics` 是 camera-to-world 的 `position`、xyzw 的 `quaternion_xyzw`，以及 `coordinate_system`：`rub`（右/上/后，镜头看 −Z）或 `rdf`（右/下/前，即 OpenCV）。`test` 视图必须带 `extrinsics`；输入视图要么全带、要么全不带 |
| `depth` | 否 | `{file, encoding, format, z_min?, z_max?}` | `linear`：EXR 或 `.npy` 浮点 z 深度；`log`：PNG 或 EXR，必须带 `0 < z_min < z_max`，按 z = z_min·(z_max/z_min)^(1−t) 解码（t 是归一化到 [0,1] 的像素值）。NaN 或 ≤ 0 表示无几何 |
| `confidence` | 否 | PNG | 8 位按 /255、16 位按 /65535 归一化到 [0,1]。没有这一项时按全 1 处理 |

**与 Atlas scene 的对应**：`make_manifest atlas` 只取 `status: active` 的视图。`camera` 和 `depth` 原样照抄 `atlas-scene.json`，各文件的 sha256 和字节数也按 JSON 里的记录核对，对不上就失败（`--no-verify` 跳过核对）。

**留出规则**：

- create 视图在前、generate 视图在后，各自保持 JSON 里的顺序；
- create 视图留出最后 `--test-create` 张作 test，缺省 1 张；只有一张 create 时缺省 0 张。第一张永远是输入；
- generate 视图每 `--test-every` 张留 1 张，缺省 4；
- 其余都是 `input`。

输入视图的顺序就是重建时的顺序，`inputs=N` 取前 N 个，所以「只用 create 视图」就是 `inputs=create 张数`。

## 环境

v1 和 v2 各用自己的环境，按各自的 `INSTALL.md` 装好。评测依赖装进 **v2** 的环境：

```bash
pip install -r manifold/requirements-eval.txt    # gsplat、lpips、opencv、plyfile、pytest
```

v1 只在单图那几组里用到：脚本通过 `--v1-python` 在 v1 的环境里调用 `InfiniSplat_v1` 的 `python -m src.demo.infer_batch_images`。所有命令都在仓库根目录下执行。

## 跑法

```bash
# 1. 生成 manifest（Atlas 导出，或者普通照片）
python -m manifold.atlas_eval.make_manifest atlas manifold/data/S1_living_room
python -m manifold.atlas_eval.make_manifest images manifold/data/my_photos

# 2. 先列出实验矩阵（不加载模型），再真正跑
python -m manifold.atlas_eval.run --scene manifold/data/S1_living_room --scene manifold/data/S2_bedroom --out manifold/runs \
    --arm I1 --arm I1-v5 --arm I1-v11 --arm I2 --arm I3 --arm S-rgb --arm S-depth \
    --model weights/da3 --v1-python /path/to/v1/env/bin/python --dry-run
#    去掉 --dry-run 即开跑。同一进程里 v2 的权重只加载一次

# 3. 外部 splat 作对照（例如面元基线；它的世界系是 Atlas 世界取 (x, −y, −z) 时用 --frame flip-yz）
python -m manifold.atlas_eval.import_splat --scene manifold/data/S1_living_room --ply surfel/scene.ply \
    --out manifold/runs --arm B0 --frame flip-yz --inputs v_... v_... --note "surfel, default params"

# 4. 对齐、渲染、打分（需要 CUDA）
python -m manifold.atlas_eval.render_eval --scene manifold/data/S1_living_room manifold/runs/S1_living_room/*/

# 5. 汇总
python -m manifold.atlas_eval.summarize manifold/runs --baseline B0 --scenes-root manifold/data
```

`--arm` 不给时为 `auto`：输入视图 ≥ 2 张跑 `I1`；只有 1 张跑 `S-rgb`，有深度时再加 `S-depth`。预设组：

| 组 | 后端 | 方式 | 说明 |
|---|---|---|---|
| `I1` | v2 | `provided`，518 | 全部输入视图加给定相机 |
| `I1-v5` / `I1-v11` | v2 | `provided`，518 | 只取前 5 / 11 个输入视图 |
| `I2` | v2 | `provided`，770 | 分辨率消融 |
| `I3` | v2 | `da3-pose`，518 | 不用给定相机，用 DA3 估位姿 |
| `S-rgb` | v1 | `rgb` | 第一个输入视图，单图 |
| `S-depth` | v1 | `lidar` | 第一个输入视图，manifest 深度作 depth prompt |

自定义组的写法是 `NAME=MODE[:LONG_SIDE[:INPUTS]]`，`MODE` 取 `provided`、`da3-pose`、`pose-free`、`v1-rgb`、`v1-lidar` 之一，例如 `--arm X=pose-free:1022:17`。组的输出目录必须是新的；`--skip-existing` 跳过 `run.json` 里状态为 ok 的组。某一组失败时，`run.json` 记 `status: failed` 和原因，其余组照常跑。

## 输出

```text
manifold/runs/<scene>/<arm>/
├── scene.ply 或 v1/<stem>/<stem>.ply
├── run.json         # 组的设定、输入视图、帧的类型、耗时、峰值显存、高斯数、PLY 字节数、git 提交
├── alignment.json   # manifest 世界 → splat 坐标系的 Sim(3) 及其残差
├── eval.json        # 每个视图与各组 (role:origin) 的指标
└── eval/<view_id>.png
manifold/runs/summary.csv, summary.md       # 每行一个 scene × arm × 组，括号里是相对 --baseline 的差值，✓ 表示更好
manifold/runs/<scene>/sheets/<view_id>.jpg       # GT | 各组渲染
```

指标：

- **画质**：PSNR、SSIM、LPIPS（缺省 alex，`--lpips-net vgg` 可换）；`psnr_masked` 和 `ssim_masked` 只算深度有效（且过置信度阈值）的像素；
- **完整度**：`coverage`，alpha > 0.5 的像素占比；
- **几何**：`depth_absrel` 和 `depth_delta1`，渲染深度换算回 manifest 世界单位后与 manifest 深度比；`depth_coverage` 是有效深度像素里被渲染覆盖的比例。

## 坐标与对齐

- **v2**：PLY 和 `cameras.npz` 在 InfiniSplat 自己的 first-camera-relative 归一化空间里。用输出相机与 manifest 输入相机拟合 Sim(3)：旋转取各相机旋转的弦平均，尺度和平移用相机中心做最小二乘。两张图、共线的相机也能拟合；所有中心重合（纯旋转）时尺度无法确定，改为在第一个输入视图上按渲染深度与 manifest 深度的中位比定。`center_rmse_relative` 在 `provided` 下应接近 0；在 `da3-pose` 下它就是位姿误差，按场景尺度归一。
- **v1**：输入相机就是 splat 坐标系的单位阵。`S-depth` 把深度缩放进 v1 接受的 (1, 100) 区间（缩放因子在 log 空间居中，记进 `run.json`），这个因子就是 splat 的尺度；`S-rgb` 的尺度由渲染深度的中位比确定。v1 的 depth prompt 采样原本没有设随机种子，这里固定为 0。
- **外部 splat**：`--frame world` 表示 PLY 就在 manifest 世界里；`--frame flip-yz` 表示 PLY 是 manifest 世界取 (x, −y, −z)。
- 第一个输入视图带深度时，每一组都会在它上面做 `depth_scale_check`：渲染深度除以 manifest 深度的中位数。它应当等于对齐得到的尺度，偏差大就说明深度与相机对不上。

## 测试与已知限制

```bash
python -m pytest manifold/tests -q     # 纯 CPU：manifest、相机换算、留出规则、对齐、深度解码、PLY 读取、汇总
```

- 真正的 v1/v2 推理、gsplat 渲染和 LPIPS **只能在有 CUDA 的机器上跑**，单测覆盖不到，第一次跑时要盯着看。
- log 深度的解码公式是暂定读法（Atlas 文档没写），改公式只需改 `manifest.log_depth` 一处。
- T_gen（`test:generate`）的 GT 本身是 Atlas 生成的，衡量的是与 Atlas 的一致性；以 `test:create` 为主。
- InfiniSplat 代码与权重的许可是 PRL v1.0：机构内部使用（包括研发和测试）要先登记，见仓库根目录的 `LICENSE`。
