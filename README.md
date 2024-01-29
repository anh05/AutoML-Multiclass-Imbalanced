#Reproduce Experiment scripts


## 1- Install relevant libraries :
```r
pip install -r requirements.txt
```

## 2- Run experiment:

```r
python [Optimization algorithm] [DataName] [seed] [Performance metric] [Classification Strategy] [Test Case]
```
### Input Parameters: 
- [Optimization algorithm]: you can choose one of `TPE.py`, `BO4ML.py`, or `DACOpt.py`
- [DataName]: `Top` for SIS top-side dataset and `Bot` SIS bottom-side dataset
- [seed]: Random seed; please use a different number for each run
- [Performance metric]: please use `gmm1` for Geometric mean micro
- [Classification Strategy]: you can choose one of `direct` ,`ovo`, or `ovr`
- [Test Case]: please choose one in the range 1 to 10

Once the optimization is completed, the final results will automatically save to `RESULTS/[Optimization algorithm].csv` and the corresponding logs at `RESULTS/LOGS`

