# Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization

> Nayeong Kim, Juwon Kang, Sungsoo Ahn, Jungseul Ok and Suha Kwak
>
> [Improving Robustness to Multiple Spurious Correlations by Multi-Objective Optimization](https://openreview.net/pdf/9f78a2f076c28d47f30480231e6908378b569466.pdf)


The experiments use the following datasets:
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [UrbanCars](https://github.com/facebookresearch/Whac-A-Mole)
- [Multi-Color MNIST](https://github.com/jayaneetha/colorized-MNIST)
- [Waterbirds](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz)
- [bFFHQ](https://drive.google.com/file/d/1Y4y4vYz6sRJRqS9jJyD06cUSR618g0Rp/view?usp=drive_link)


# Prerequisites
See requirements.yml

# Datasets and code

To run our code, you will need to change the `root_dir` variable in `data/data.py`.
- The main point of entry to the code is `run_expt.py`.

- The code for evaluating model checkpoint is `eval_checkpoint.py`.

### Datasets

Our code expects the following files/folders for each dataset:

You can download csv files from [this Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset) or [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). 
```
CelebA
 └ list_eval_partition.csv
 └ list_attr_celeba.csv
 └ img_align_celeba/
```



```
urbancars
 └ bg-0.5_co_occur_obj-0.5
 └ bg-0.95_co_occur_obj-0.95
```

```
multi_color_mnist
 └ ColoredMNIST-SkewedA0.01-SkewedB0.05-Severity4
```

```
waterbirds
 └ waterbird_complete95_forest2water2
```

```
bffhq
 └ 0.5pct
 └ valid
 └ test
```

## Model checkpoints
You can download model checkpoints from [here]().
```
best_epoch
 └ MultiCelebA_best_epoch.pth
 └ UrbanCars_best_epoch.pth
 └ MultiMNIST_best_epoch.pth
 └ Waterbirds_best_epoch.pth
 └ CelebA_best_epoch.pth
 └ bFFHQ_best_epoch.pth
```

# A sample command 

### Evaluation
Two-bias setting of MultiCelebA 

`python eval_checkpoint.py --dataset=MultiCelebA --target_name=High_Cheekbones --n_epoch=1 --shift_type=confounder`

Three-bias setting of MultiCelebA 

`python eval_checkpoint.py --dataset=MultiCelebA --target_name=High_Cheekbones_3types --n_epoch=1 --shift_type=confounder`

UrbanCars

`python eval_checkpoint.py --dataset=UrbanCars --target_name=cars --n_epoch=1 --shift_type=confounder`

Multi-Color MNIST

`python eval_checkpoint.py --dataset=MultiMNIST --target_name=digits --n_epoch=1 --shift_type=confounder`

Waterbirds

`python eval_checkpoint.py --dataset=Waterbirds --target_name=birds --n_epoch=1 --shift_type=confounder`

CelebA

`python eval_checkpoint.py --dataset=CelebA --target_name=Blond_Hair --n_epoch=1 --shift_type=confounder`

bFFHQ

`python eval_checkpoint.py --dataset=bFFHQ --target_name=class --n_epoch=1 --shift_type=confounder`


### Training

MultiCelebA

`python run_expt.py -s confounder -d MultiCelebA -t High_Cheekbones -c Young Male --batch_size 512 --lr 0.0002 --weight_decay 1 --reweight_groups --model resnet18 --n_epochs 60 --show_progress --alpha_lr 0.01 --alpha_step 10 --optimizer SGD --seed 64 --log_every 10 --wandb`

UrbanCars

`python run_expt.py --batch_size=128 --alpha_lr=0.001 --alpha_step=10 --dataset=UrbanCars --early_stop_v=50 --log_every=10 --lr=0.01 --model=resnet50 --momentum=0 --n_epochs=300 --optimizer=SGD --shift_type=confounder --target_name=Cars_crop --weight_decay=0.1 --reweight_groups --seed 0 --wandb`

Multi-Color MNIST

`python run_expt.py -s confounder -d MultiMNIST -t digit -c L R --batch_size 512 --lr 0.02 --weight_decay 0.0001 --reweight_groups --model MLP --n_epochs 60 --show_progress --alpha_lr 0.002 --alpha_step 50 --optimizer Adam --seed 1 --log_every 10 --wandb`

Waterbirds

`python run_expt.py -s confounder -d Waterbirds -t birds -c background --batch_size 128 --lr 0.0001 --weight_decay 0.1 --reweight_groups --model resnet50 --n_epochs 400 --show_progress --alpha_lr 0.001 --alpha_step 5 --optimizer SGD --seed 0 --log_every 10 --wandb`

CelebA

`python run_expt.py -s confounder -d CelebA -t Blond_Hair -c background --batch_size 128 --lr 0.002 --weight_decay 0.00001 --reweight_groups --model resnet50 --n_epochs 20 --show_progress --alpha_lr 0.0001 --alpha_step 1 --optimizer Adam --seed 64 --log_every 10 --wandb`

bFFHQ

`python run_expt.py -s confounder -d bFFHQ -t class -c bias1 --batch_size 64 --lr 0.0005 --weight_decay 0.0001 --reweight_groups --model resnet18 --n_epochs 200 --show_progress --alpha_lr 0.0005 --alpha_step 10 --optimizer Adam --seed 0 --log_every 10 --wandb`

