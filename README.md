# Source-Free Unsupervised Domain Adaptation in Imbalanced Datasets

## Notes

Libraries such as torch, numpy, tensorboardX are required to run the code. Please install the needed libraries according to the runtime prompts.

We used Python 3.6 to implement the experiments.

Datasets SVHN, MNIST, and USPS can be downloaded automatically.

Dataset VisDA-C can be downloaded from [train.tar](https://drive.google.com/file/d/0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA/view?usp=sharing) and [validation.tar](https://drive.google.com/file/d/0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA/view?usp=sharing). You can refer to the source code of [SHOT](https://github.com/tim-learn/SHOT) to figure out how to generate txt files for the training.



## Running Codes

First, cd to the corresponding folder, then run

### Digit

``` python
python new_uda_digit.py --dset s2m --balance False # SHOT
python new_uda_digit.py --dset s2m --balance True --naive_dis True --naive_sample True # NN
python new_uda_digit.py --dset s2m --balance True --naive_dis True --naive_sample False --threshold 0.3 # NP0.3
python new_uda_digit.py --dset s2m --balance True --naive_dis False --naive_sample True # PN
python new_uda_digit.py --dset s2m --balance True --naive_dis False --naive_sample False --threshold 0.3 # PP0.3
python new_uda_digit.py --dset s2m --balance True --naive_dis False --naive_sample False --threshold_speed linear --threshold 0.3 --max_threshold 1 # PP0.3->1, alpha=1
```

### Object

```python
python image_source.py # Train the source model
python new_image_target.py --balance False # SHOT
python new_image_target.py --balance True --naive_dis True --naive_sample True # NN
python new_image_target.py --balance True --naive_dis True --naive_sample False --threshold 0.3 # NP0.3
python new_image_target.py --balance True --naive_dis False --naive_sample True # PN
python new_image_target.py --balance True --naive_dis False --naive_sample False --threshold 0.3 # PP0.3
python new_image_target.py --balance True --naive_dis False --naive_sample False --threshold_speed linear --threshold 0.3 --max_threshold 1 # PP0.3->1, alpha=1
```

