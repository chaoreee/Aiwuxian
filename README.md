# Round 2 Physical-AI 无线信道生成

本项目面向 `Problem-Round2`：根据三维位置与场景地图生成
`(256, 4, 192)` 的复数 MIMO-OFDM 信道。

当前方案不是射线追踪。模型将位置、服务基站和物理相对坐标编码为隐变量，再由
3D 卷积神经解码器生成稀疏的角度–时延信道；额外的神经头负责预测覆盖概率和
路径损耗。最终通过可逆 FFT 还原复数频域信道。

## 已确认的数据特性

- 两个覆盖区各有 2000 个训练点、250 个测试点。
- 测试点是成片空间空洞，不适合用随机切分估计榜单性能。
- 训练集中有 262 个全零信道，必须把覆盖判别作为显式学习任务。
- 信道功率跨越多个数量级，所有指标都必须用 float64 累加。
- 角度–时延域前 16 个 delay bins 保留了绝大部分可预测能量。

## 快速开始

```powershell
python prepare_cache.py --data-dir Round2_Map --cache-dir cache --delay-bins 16
python train.py --data-dir Round2_Map --cache-dir cache --output-dir outputs_final `
  --epochs 6 --scheduler-tmax 20 --batch-size 32 --lr 0.0003 `
  --width 256 --fourier-levels 6 --pas-mode both --train-all
python predict_hybrid.py --data-dir Round2_Map --cache-dir cache `
  --checkpoint outputs_final/best.pt --pas-layout upa `
  --output outputs_final/Round2_Test_Channel.npy
```

预测文件默认写入 `outputs_final/Round2_Test_Channel.npy`。

## 本地空间块验证

验证集由 12 个连通空间空洞组成，其测试点到最近训练点的距离分布与真实测试集接近。
完整 192 子载波的稳定 float64 评测结果如下（NMSE 约为 1）：

| 方案 | UPA-PAS 假设 | 一维端口 PAS 假设 |
|---|---:|---:|
| 1-NN | 0.5283 | 0.5247 |
| 平滑神经场 | 0.5566 | 0.5590 |
| AI + 邻域谱混合（UPA恢复） | **0.5913** | 0.5738 |
| AI + 邻域谱混合（一维恢复） | 0.5755 | **0.5932** |

`outputs_final/Round2_Test_Channel_UPA.npy` 是默认候选；若线上反馈确认 PAS 是对扁平
256 端口直接做 DFT，则改用 `Round2_Test_Channel_FLAT.npy`。
