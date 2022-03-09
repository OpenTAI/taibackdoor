## Introduction

TAIBackdoor is an open adversarial machine learning freamework based on PyTorch. It is
a part of the [OpenTAI](https://github.com/OpenTAI) project.

- **Modular Design**  We decompose the adversarial machine learning framework into different components, and one can easily construct a customized project by combining different modules.

- **Designed for Research** We aim at providing highly flexible modules for adversarial machine learning researchers.

- **State of the art** We provide implementations of state-of-the-art attack/defence techniques published in different venues.

- **Flexibility** Our framework provide flexible modules that can be integrated with other adversarial ML frameworks such as [RobustBench](https://github.com/RobustBench/robustbench)

## Overview of this project
- **attacks:** implementations of backdoor attacks
- **defenses:** implementations of backdoor defenses
- **datasets:** implementation of wrapper for commonly used dataset based on torchvision
- **losses:** implementations for attacks/defenses training losses
- **models:** implementations for commonly used models
- **training** implementations of training pipeline


## Contributing to TAIBackdoor

We appreciate all contributions to improve for TAIBackdoor. Welcome community users to participate in our projects. Please refer to [CONTRIBUTING.md](https://github.com/OpenTAI/taiadv/blob/main/CONTRIBUTING.md) for guideline.

## Acknowledgement
TAIAdv is an open-source project that is contributed by researchers from the community. Part of the code is based on existing papers, either reimplementation or open-source code provided by authors. For complete list of paper, please see [ACKNOWLEDGEMENT.md](https://github.com/OpenTAI/taiadv/blob/main/ACKNOWLEDGEMENT.md)

## Other Projects in OpenTAI
- [TAIXIA](https://github.com/OpenTAI/taixai): Explainable AI Toolbox
- [TAICorruption](https://github.com/OpenTAI/taicorruption): Common Corruption Robustness Toolbox and Benchmark
- [TAIBackdoor](https://github.com/OpenTAI/taibackdoor): Backdoor Attack and Defense Toolbox and Benchmark
- [TAIFairness](https://github.com/OpenTAI/taifairness): AI Fairness Toolbox and Benchmark
- [TAIPrivacy](https://github.com/OpenTAI/taiprivacy): Privacy Attack and Defense Toolbox and Benchmark
- [TAIIP](https://github.com/OpenTAI/taiip): AI Intellectual Property Protection Toolbox and Benchmark
- [TAIDeepfake](https://github.com/OpenTAI/taideepfake): Deepfake Detection Toolbox and Benchmark
