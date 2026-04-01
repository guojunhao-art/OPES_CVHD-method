# OPES_CVHD 实现快速熟悉笔记

这个仓库是将 **OPES** 与 **CVHD** 结合到 PLUMED/LAMMPS 工作流中的参考实现，核心代码在 `src/`，示例输入在 `examples/`。

## 1) 三个核心源码文件

- `src/OPESmetadCV.cpp`
  - 基于 PLUMED 的 OPES 模块扩展得到。
  - 负责构建 OPES 偏置，并在此实现中加入了与 CVHD 兼容的逻辑（如和事件/加速相关的参数与状态）。

- `src/CVHDM.cpp`
  - 实现 `CVHDM` 函数：将一个或多个 CV 通过 cutoff 与 p-norm 组合成用于 CVHD 的“夹紧”变量。
  - 支持多参数输入（`ARG`、`CUTOFF`、`P`），可把多类畸变信息合成为单一驱动变量。

- `src/GlobalDistortion.cpp`
  - 实现 `GLOBALDISTORTION` CV：在原子对参考列表基础上，计算全局键畸变。
  - 支持 reset 参考列表、cutoff 过滤、双原子组定义、可选 switching function 等。

## 2) 示例目录怎么用

- `examples/cu_vacancy/`：铜空位体系，包含 `CVHD` 与 `OPES_CVHD` 两套输入，便于做对照。
- `examples/MCH_py/`：甲基环己烷热解相关体系，同样有 `CVHD` 与 `OPES_CVHD` 对照输入。
- `examples/styryl/`：另一个可直接运行的输入示例。

建议优先看每个示例目录里的 `plumed.dat`，快速对应“CV 定义 → 偏置设置 → 输出项”。

## 3) 推荐阅读顺序（最省时间）

1. `README.md`：方法概览、代码来源、示例结构。
2. `examples/*/OPES_CVHD/plumed.dat`：先看用户输入层怎么调用。
3. `src/CVHDM.cpp` 与 `src/GlobalDistortion.cpp`：理解 CV 构造与事件触发基础。
4. `src/OPESmetadCV.cpp`：最后看 OPES 偏置与 CVHD 结合细节。

## 4) 初步结论

- 这是一个偏“论文复现/方法验证”风格的仓库，不是完整打包发行版。
- 你后续如果要把它用于新体系，主要工作会集中在：
  - 设计合适的 `GLOBALDISTORTION` / `CVHDM` 参数；
  - 设定 OPES 偏置参数（如 `PACE`、`BARRIER` 等）；
  - 用已有示例先做最小可跑通验证。
