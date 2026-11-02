# D7-case2 supplemental assets

## 补充 2 完成核验

13 个案例的原生 0.02 m/像素裁剪底图均已生成，原 0.1 m 底图与裁剪变换保留。47 张独立的视野外目标点 other 视角图片均为 640×480，并标记 world_target_v2；HFOV 内目标不另渲染，复用前视图。19 个关键帧槽位标记 same_place=false。所有关键帧裁剪像素坐标经反变换均还原原世界坐标（误差 <1e-6 m）。仅进行了 CPU 完整性核验，未排版；此检查不等价于逐图人工视觉审校。

13 个已有案例；D 的 30 条补录目录未交回，中段岔路仍缺 2 个。本次不排版、不修改既有图片。

other 使用另一模型同 step 的目标世界坐标投影，不进行候选匹配；STOP 取实际停止位置。同 step 不存在则 null。两模型位置距离 >1.5 m 标 same_place=false，不因此丢弃目标。on_GT 使用到 GT 三维折线距离 ≤1 m 的几何近似。

画面像素是目标点透视投影，不是物体检测框。other 在 HFOV 内复用原前视图，HFOV 外渲染 step*_other.png；原候选匹配版另存 *_other_matched.png。0.02 m/像素裁剪图由 navmesh 重新采样，非图像放大；原0.1m版本保留。

早停测地距离使用实际终止轨迹位置，而非 STOP 前可能尚未 teleport 的决策位置。Habitat-Sim 使用 ShortestPath + find_path（等价的 navmesh 测地查询），无欧氏替代。

## User command

```bash
CUDA_VISIBLE_DEVICES=5 /home/xukai/anaconda3/envs/etpr1-py38/bin/python /home/xukai/code/ETP-R1-snapshot/ETP-R1/scripts/eval_d7/case2/render_views.py --supplement-only
```

完成后运行 report_assets.py 更新此清单；不运行 compose.py。

## Episode 1033

[Manifest](figs/d7_case2/1033/manifest.json) · [Summary](figs/d7_case2/1033/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step8 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step3 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step8 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step2_fpv.png (existing) | 640×480 | 齐全 |
| D/step2_other.png (new) | 640×480 | 齐全 |
| D/step2_pano.png (existing) | 7680×480 | 齐全 |
| D/step2_sel.png (new) | 640×480 | 齐全 |
| D/step3_fpv.png (existing) | 640×480 | 齐全 |
| D/step3_pano.png (existing) | 7680×480 | 齐全 |
| D/step3_sel.png (new) | 640×480 | 齐全 |
| D/step8_fpv.png (existing) | 640×480 | 齐全 |
| D/step8_other.png (new) | 640×480 | 齐全 |
| D/step8_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step2_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step2_other.png (new) | 640×480 | 齐全 |
| baseline/step2_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step2_sel.png (new) | 640×480 | 齐全 |
| baseline/step3_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step3_other.png (new) | 640×480 | 齐全 |
| baseline/step3_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step3_sel.png (new) | 640×480 | 齐全 |
| baseline/step8_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step8_other.png (new) | 640×480 | 齐全 |
| baseline/step8_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 224×133 | 齐全 |
| topdown_crop.png (0.1m preserved) | 153×112 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 760×560 | 齐全 |

## Episode 1105

[Manifest](figs/d7_case2/1105/manifest.json) · [Summary](figs/d7_case2/1105/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step14 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step5 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_other.png (new) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step14_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step14_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 293×450 | 齐全 |
| topdown_crop.png (0.1m preserved) | 102×270 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 503×1342 | 齐全 |

## Episode 1134

[Manifest](figs/d7_case2/1134/manifest.json) · [Summary](figs/d7_case2/1134/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step9 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step7 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step7_fpv.png (existing) | 640×480 | 齐全 |
| D/step7_other.png (new) | 640×480 | 齐全 |
| D/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step9_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step9_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 430×284 | 齐全 |
| topdown_crop.png (0.1m preserved) | 201×129 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 1004×643 | 齐全 |

## Episode 1225

[Manifest](figs/d7_case2/1225/manifest.json) · [Summary](figs/d7_case2/1225/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step7 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step5 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_other.png (new) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step1_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step1_other.png (new) | 640×480 | 齐全 |
| baseline/step1_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step1_sel.png (new) | 640×480 | 齐全 |
| baseline/step7_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step7_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 430×284 | 齐全 |
| topdown_crop.png (0.1m preserved) | 253×135 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 1264×673 | 齐全 |

## Episode 1263

[Manifest](figs/d7_case2/1263/manifest.json) · [Summary](figs/d7_case2/1263/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step3 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step4 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step4 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step3 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step5 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step3_fpv.png (existing) | 640×480 | 齐全 |
| D/step3_pano.png (existing) | 7680×480 | 齐全 |
| D/step3_sel.png (new) | 640×480 | 齐全 |
| D/step4_fpv.png (existing) | 640×480 | 齐全 |
| D/step4_other.png (new) | 640×480 | 齐全 |
| D/step4_pano.png (existing) | 7680×480 | 齐全 |
| D/step4_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step3_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step3_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step3_sel.png (new) | 640×480 | 齐全 |
| baseline/step4_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step4_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 224×133 | 齐全 |
| topdown_crop.png (0.1m preserved) | 150×86 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 746×426 | 齐全 |

## Episode 1706

[Manifest](figs/d7_case2/1706/manifest.json) · [Summary](figs/d7_case2/1706/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step8 投影 | — | same_place=False; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step8 投影 | — | same_place=False; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step5 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step8 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| D/step5_sel.png (new) | 640×480 | 齐全 |
| D/step6_fpv.png (existing) | 640×480 | 齐全 |
| D/step6_other.png (new) | 640×480 | 齐全 |
| D/step6_pano.png (existing) | 7680×480 | 齐全 |
| D/step6_sel.png (new) | 640×480 | 齐全 |
| D/step8_fpv.png (existing) | 640×480 | 齐全 |
| D/step8_other.png (new) | 640×480 | 齐全 |
| D/step8_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step7_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step7_other.png (new) | 640×480 | 齐全 |
| baseline/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step7_sel.png (new) | 640×480 | 齐全 |
| baseline/step8_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step8_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 293×450 | 齐全 |
| topdown_crop.png (0.1m preserved) | 181×107 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 904×529 | 齐全 |

## Episode 1709

[Manifest](figs/d7_case2/1709/manifest.json) · [Summary](figs/d7_case2/1709/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step5 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step6 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step6 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step5 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step8 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| D/step5_sel.png (new) | 640×480 | 齐全 |
| D/step6_fpv.png (existing) | 640×480 | 齐全 |
| D/step6_other.png (new) | 640×480 | 齐全 |
| D/step6_pano.png (existing) | 7680×480 | 齐全 |
| D/step6_sel.png (new) | 640×480 | 齐全 |
| D/step8_fpv.png (existing) | 640×480 | 齐全 |
| D/step8_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step5_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step5_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step5_sel.png (new) | 640×480 | 齐全 |
| baseline/step6_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step6_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 155×156 | 齐全 |
| topdown_crop.png (0.1m preserved) | 135×79 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 672×391 | 齐全 |

## Episode 1813

[Manifest](figs/d7_case2/1813/manifest.json) · [Summary](figs/d7_case2/1813/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step7 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step8 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step8 投影 | — | same_place=True; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step7 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step9 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step7_fpv.png (existing) | 640×480 | 齐全 |
| D/step7_pano.png (existing) | 7680×480 | 齐全 |
| D/step7_sel.png (new) | 640×480 | 齐全 |
| D/step8_fpv.png (existing) | 640×480 | 齐全 |
| D/step8_other.png (new) | 640×480 | 齐全 |
| D/step8_pano.png (existing) | 7680×480 | 齐全 |
| D/step8_sel.png (new) | 640×480 | 齐全 |
| D/step9_fpv.png (existing) | 640×480 | 齐全 |
| D/step9_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step7_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step7_sel.png (new) | 640×480 | 齐全 |
| baseline/step8_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step8_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 218×375 | 齐全 |
| topdown_crop.png (0.1m preserved) | 87×211 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 430×1047 | 齐全 |

## Episode 302

[Manifest](figs/d7_case2/302/manifest.json) · [Summary](figs/d7_case2/302/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step3 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step14 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step3 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step4 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step6 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step3_fpv.png (existing) | 640×480 | 齐全 |
| D/step3_pano.png (existing) | 7680×480 | 齐全 |
| D/step3_sel.png (new) | 640×480 | 齐全 |
| D/step4_fpv.png (existing) | 640×480 | 齐全 |
| D/step4_pano.png (existing) | 7680×480 | 齐全 |
| D/step4_sel.png (new) | 640×480 | 齐全 |
| D/step6_fpv.png (existing) | 640×480 | 齐全 |
| D/step6_other.png (new) | 640×480 | 齐全 |
| D/step6_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step14_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step14_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step3_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step3_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step3_sel.png (new) | 640×480 | 齐全 |
| baseline/step4_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step4_other.png (new) | 640×480 | 齐全 |
| baseline/step4_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step4_sel.png (new) | 640×480 | 齐全 |
| topdown.png (existing) | 224×133 | 齐全 |
| topdown_crop.png (0.1m preserved) | 187×125 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 935×620 | 齐全 |

## Episode 740

[Manifest](figs/d7_case2/740/manifest.json) · [Summary](figs/d7_case2/740/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step7 投影 | — | same_place=False; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step8 投影 | — | same_place=False; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step8 投影 | — | same_place=False; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step6 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step9 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step6_fpv.png (existing) | 640×480 | 齐全 |
| D/step6_pano.png (existing) | 7680×480 | 齐全 |
| D/step6_sel.png (new) | 640×480 | 齐全 |
| D/step7_fpv.png (existing) | 640×480 | 齐全 |
| D/step7_other.png (new) | 640×480 | 齐全 |
| D/step7_pano.png (existing) | 7680×480 | 齐全 |
| D/step7_sel.png (new) | 640×480 | 齐全 |
| D/step9_fpv.png (existing) | 640×480 | 齐全 |
| D/step9_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step7_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step7_sel.png (new) | 640×480 | 齐全 |
| baseline/step8_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step8_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 293×450 | 齐全 |
| topdown_crop.png (0.1m preserved) | 147×132 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 736×653 | 齐全 |

## Episode 872

[Manifest](figs/d7_case2/872/manifest.json) · [Summary](figs/d7_case2/872/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step6 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step7 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step1_fpv.png (existing) | 640×480 | 齐全 |
| D/step1_other.png (new) | 640×480 | 齐全 |
| D/step1_pano.png (existing) | 7680×480 | 齐全 |
| D/step1_sel.png (new) | 640×480 | 齐全 |
| D/step7_fpv.png (existing) | 640×480 | 齐全 |
| D/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step6_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step6_other.png (new) | 640×480 | 齐全 |
| baseline/step6_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 430×284 | 齐全 |
| topdown_crop.png (0.1m preserved) | 314×101 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 1570×505 | 齐全 |

## Episode 906

[Manifest](figs/d7_case2/906/manifest.json) · [Summary](figs/d7_case2/906/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step8 投影 | — | same_place=False; other_in_fov=True; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step8 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step8_fpv.png (existing) | 640×480 | 齐全 |
| D/step8_other.png (new) | 640×480 | 齐全 |
| D/step8_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step8_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step8_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 430×284 | 齐全 |
| topdown_crop.png (0.1m preserved) | 214×122 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 1067×607 | 齐全 |

## Episode 978

[Manifest](figs/d7_case2/978/manifest.json) · [Summary](figs/d7_case2/978/summary.txt)

| 文件 | 尺寸（宽×高） | 状态 |
|---|---|---|
| baseline/step4 投影 | — | same_place=True; other_in_fov=True; sel_null=False; other_null=True（STOP/无同 step/HFOV内无需新图） |
| baseline/step6 投影 | — | same_place=False; other_in_fov=False; sel_null=True; other_null=False（STOP/无同 step/HFOV内无需新图） |
| D/step7 投影 | — | same_place=None; other_in_fov=None; sel_null=True; other_null=True（STOP/无同 step/HFOV内无需新图） |
| D/step0_fpv.png (existing) | 640×480 | 齐全 |
| D/step0_other.png (new) | 640×480 | 齐全 |
| D/step0_pano.png (existing) | 7680×480 | 齐全 |
| D/step0_sel.png (new) | 640×480 | 齐全 |
| D/step4_fpv.png (existing) | 640×480 | 齐全 |
| D/step4_other.png (new) | 640×480 | 齐全 |
| D/step4_pano.png (existing) | 7680×480 | 齐全 |
| D/step4_sel.png (new) | 640×480 | 齐全 |
| D/step5_fpv.png (existing) | 640×480 | 齐全 |
| D/step5_other.png (new) | 640×480 | 齐全 |
| D/step5_pano.png (existing) | 7680×480 | 齐全 |
| D/step5_sel.png (new) | 640×480 | 齐全 |
| D/step7_fpv.png (existing) | 640×480 | 齐全 |
| D/step7_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step0_other.png (new) | 640×480 | 齐全 |
| baseline/step0_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step0_sel.png (new) | 640×480 | 齐全 |
| baseline/step4_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step4_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step4_sel.png (new) | 640×480 | 齐全 |
| baseline/step5_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step5_other.png (new) | 640×480 | 齐全 |
| baseline/step5_pano.png (existing) | 7680×480 | 齐全 |
| baseline/step5_sel.png (new) | 640×480 | 齐全 |
| baseline/step6_fpv.png (existing) | 640×480 | 齐全 |
| baseline/step6_other.png (new) | 640×480 | 齐全 |
| baseline/step6_pano.png (existing) | 7680×480 | 齐全 |
| topdown.png (existing) | 293×450 | 齐全 |
| topdown_crop.png (0.1m preserved) | 110×109 | 齐全 |
| topdown_crop_002.png (0.02m rerender) | 547×538 | 齐全 |
