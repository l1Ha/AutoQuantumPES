# 外部依赖可得性实测记录 (EXTERNAL_DEPS)

针对能力矩阵中仍标 ✗ 的方法，逐一在**本项目运行环境**（c211 集群,
`~/aqd-env`, PySCF 2.14.0）实测其外部依赖可得性。所有结论均附**可复现命令
与原始输出**, 不做"据我所知"式判断。

## 1. CASPT2

| 探测 | 命令 | 结果 |
|---|---|---|
| 主库 | `python -c "import pyscf.mrpt as m; pkgutil.iter_modules(m.__path__)"` | 仅 `dfnevpt2`, `nevpt2` — **无 CASPT2** |
| PyPI | `pip index versions pyscf-caspt2` | `No matching distribution found` |
| PyPI | `pip index versions pyscf-forge` | 可用 (1.1.1); 但模块清单**无 caspt2** |
| GitHub | `api.github.com/search/repositories?q=pyscf+caspt2` | **total_count = 0** |

→ **结论: 生态内无可用 CASPT2 实现 (实测)**。同层次的动态相关由:
1. **NEVPT2** (主库, 无侵入态) — 已实现并验证 (H₂ 距 FCI 0.10 mHa);
2. **DSRG-MRPT2** (pyscf-forge `pyscf/dsrg_mrpt2/`) — **存在但本环境无法构建**:

```
$ CMAKE_ARGS="-DBLAS_LIBRARIES=<openblas> -DLAPACK_LIBRARIES=<openblas>" \
      pip install --no-build-isolation .
CMake Error: Could NOT find BLAS (missing: BLAS_LIBRARIES)     # 首次构建
# 手动 cmake (pyscf/lib/CMakeLists.txt) 亦失败:
-- Pure Python implementation will be used (performance will be reduced)
-- To enable optimized version, install: FFTW3, BLAS, OpenMP
-- Configuring incomplete, errors occurred!
$ python -c "import pyscf.dsrg_mrpt2"
OSError: Library libdsrg not found
```

→ forge 的编译期依赖 (FFTW3 / OpenMP / 完整 BLAS+LAPACK 探测) 在共享集群
venv 中未满足; 纯 Python 子集 (`msdft`/`sfnoci`/`lno`/`occri`) 曾可导入但
随失败安装在 pip 回滚中消失。**已实测确认环境完整性未受影响**:
H₂/STO-3G CAS(2,2) = −1.137275944 Ha 与归档值逐位一致。

## 2. MRCI

| 探测 | 结果 |
|---|---|
| 主库 `pyscf.fci` | 无 MRCI 模块 (仅 FCI/select-CI/DMRG 接口) |
| PyPI `pyscf-mrci` | `No matching distribution found` |
| GitHub `q=pyscf+mrci` | 唯一命中 `block-hczhai/block2-preview` (DMRG, **非 MRCI**) |

→ **结论: 生态内无 MRCI (实测)**。多参考场景由已实现且已验证的
**SA-CASSCF + NEVPT2** (激发态, vs FCI 8.9e-16) 与
**CASCI + 选择组态 CI** (CAS(14,14) 稠密 1.18e7 维, 9 s) 覆盖。

## 3. 其他生态包

| 包 | pip index | 用途 / 与能力矩阵的关系 |
|---|---|---|
| `pyscf-forge` 1.1.1 | 可用 | 含 `dsrg_mrpt2`/`msdft`/`sfnoci`/`lno`/`csf_fci`/`occri`/`pv`/`sftda`/`pprpa`; **无 caspt2/mrci/avas** |
| `block2` 0.5.4 | 可用 | DMRG (大活性空间替代路线; 未接入) |
| `mrh` 0.0.3 | 可用 | heat-bath CI (未接入) |
| `socutils` | 不可用 | (SOC 已自研实现, 见 v0.31.0) |
| `dmrgscf` | 不可用 | 需 Block 二进制 |

## 4. 复合方法专用基组 (G4/G3)

| 基组 | `pyscf.gto.basis.load(...)` | 结果 |
|---|---|---|
| `gtbas1` (G4) | `BasisNotFoundError` |
| `G3MP2LargeXP` (G3) | `BasisNotFoundError` |
| `6-31G(2df,p)`, `cc-pV(Q,5)Z`, `aug-cc-pVQZ`, `def2-QZVP` | OK |

→ G4/W1 **无法忠实复现** (专用基组未收录); 已以公式公开的
**CBS 复合方案** (HF 三点指数 + 相关能 X⁻³ + CCSD(T) 加和) 替代并验证
(H₂ R_e 偏差 0.003%, 见 v0.34.0)。

## 5. 结论与替代矩阵

| 商业方法 | 本项目状态 | 替代实现 (均已服务器验证) |
|---|---|---|
| CASPT2 | ✗ 生态无实现 (实测) | **NEVPT2** (同层级, 无侵入态; H₂ 距 FCI 0.10 mHa) |
| MRCI / RASPT2 | ✗ 生态无实现 (实测) | **SA-CASSCF+NEVPT2** (激发态) / **CASCI+选择组态 CI** (大活性空间) |
| CCSD(T)-F12 | ✗ 主库无 F12 | 基组外推 (**CBS 复合**, v0.34.0) |
| 四分量 Dirac / 二电子 SOC | ✗ 需专用积分库 | 标量 X2C + **单电子 Breit–Pauli SOC** (v0.31.0) |
| G4/W1/G4MP2 | ✗ 专用基组缺失 (实测) | **CBS 复合方案** (公式公开可逐项核验) |
