# Infinite Class Mixup
This repository contains the PyTorch code for the BMVC 2023 paper "Infinite Class Mixup".
<br>
The paper is available here: https://arxiv.org/abs/2305.10293
<br><br>
The repository includes:
* Utilization files for mixing and loading datasets.
* ResNet with extra transformation on top for dual-axis mixup learning.
* Code for running on CIFAR and CUB Birds.

## Running CIFAR experiments.

To run a baseline ResNet-32 on CIFAR-10 or CIFAR-100, run the following command:
```
python mixup_cifar.py -d cifar10/100 -m none
```
Note: the base code assumes the data is stored in "../../data/". If your folder is different, change the directory accordingly in "utils.py".
<br><br>
To run standard and our Mixup variants (e.g., on CIFAR-100), run one of the following commands:
```
python mixup_cifar.py -d 100 -m mixup -a 0.2
python mixup_cifar.py -d 100 -m icmixup-f -a 0.2
python mixup_cifar.py -d 100 -m regmixup -a 20
python mixup_cifar.py -d 100 -m regicmixup-f -a 20
```
To get results using only the class- or pair-axis, replace the "-f" in the ic variants with "-s" or "-c".

## Running CUB Birds experiments.
For CUB Birds experiments, you can run the same commands as above using the "mixup_birds.py" file. Note again that the efault data directory is "../../data/" in "utils.py".

## Citing the paper.
Please cite the paper as follows:
```
@inproceedings{mensink2023infinite,
  title={Infinite Class Mixup},
  author={Mensink, Thomas and Mettes, Pascal},
  booktitle={British Machine Vision Conference},
  year={2023}
}
```
