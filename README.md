# surface_code_decoding_benchmark
Test the performance of existing surface code decoding methods under circuit-level noise models.

## 调研目前现有的量子系统

* PyMatching（BeliefMatching、STIMBPOSD）: https://github.com/oscarhiggott
* qiskit-qec: https://github.com/qiskit-community/qiskit-qec
* qtcodes: https://github.com/yaleqc/qtcodes
* mqt-qecc: https://github.com/cda-tum/mqt-qecc
* qecsim: https://github.com/qecsim/qecsim
* PanQEC: https://github.com/panqec/panqec
* FlamingPy: https://github.com/XanaduAI/flamingpy
* qsurface: https://github.com/watermarkhu/qsurface
* TensorQEC.jl: https://github.com/nzy1997/TensorQEC.jl
* PECOS: https://github.com/PECOS-packages/PECOS
* qecGPT: https://github.com/CHY-i/qecGPT
* neural-decoder: https://github.com/Krastanov/neural-decoder
* Chromobius: https://github.com/quantumlib/chromobius
* fusion-blossom: https://github.com/yuewuo/fusion-blossom

含噪模拟工具：
* stim
* qiskit-aer
* cirq

## 测试（Benchmark）

目前的量子纠错的含噪模拟过程，主要分为两个噪声模型，其中一个为现象级噪声模型，另一个为电路级噪声模型。现象级噪声模型大多在自己内部实现，比如qecsim等，在21、22年之前研究的比较多。另一个电路级噪声模型，通常目前有stim、cirq、qiskit-aer能够实现含噪模拟。由于目前考虑的QEC不存在非clifford门，所以stim对stablizer模拟进行了大量的优化，具备很好的QEC线路模拟效果。

关于量子解码算法，在最早的量子解码算法不利用任何的噪声信息，即MWPM算法。在约2013~2021年，现象级噪声模型逐渐兴起，大家开始利用现象级噪声模型得到的现象级噪声信息来实现R-MWPM、R-UF、BP、BP+OSD以及基于TN的最大似然解码方法。在2021年之后，随着stim提出以及google的Nature 文章Suppressing quantum errors by scaling a surface code logical qubit，大家开始考虑更加细致的电路级噪声信息。目前解码电路级噪声信息的方法主要有两类，一类为将电路级噪声信息，通过超图分解或者BP的策略，转换为现象级噪声信息，另一种则是基于电路级噪声信息的直接解码（部分论文尝试利用近似张量网络收缩技术来加速）。

关于测试，我们将主要测试在surface code下的电路级噪声模型的
* pymatching
* BeliefMatching
* STIMBPOSD
* online MLD

其中具体数据参考data中的数据，包括码矩为3且重复轮次为1、码矩为3且重复轮次为3, 码矩为5且重复轮次为1的方法。

## 环境配置
```
conda create -n decoding_benchmark python=3.10

python -m pip install jupyter
python -m pip install stim

python -m pip install pymatching --upgrade
python -m pip install beliefmatching
python -m pip install stimbposd
```

在调研期间，qiskit-qec的解码方式调用pymatching。