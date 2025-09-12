# How to run:

Check the respective MIMIC folder, viewing the `processed/` folder, you should find the `ehr` and `note` csv after running the below command (Windows).

Basically just a bunch of file executions found in each MIMIC folder md, you can figure out how to run them on other systems!

## MIMIC-III

```
python -m mimic3benchmark.scripts.extract_subjects D:\Lab\Research\EMERGE-REPLICATE\datasets\mimic-iii data/root/ ;
if ($?) { python -m mimic3benchmark.scripts.validate_events data/root/ } ;
if ($?) { python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/ } ;
if ($?) { python -m preprocess_mimic3 data/root/ data/processed/ } ;
if ($?) { python -m scripts.extract_notes D:\Lab\Research\EMERGE-REPLICATE\datasets\mimic-iii data/root/ } ;
if ($?) { python -m preprocess_notes data/root/ data/processed/ }
```

## MIMIC-IV

# All credits goes to the following contributors (I only made slight modifications):

- [Yinghao Zhu](https://github.com/yhzhu99)
- [Shiyun Xie](https://github.com/SYXieee)
- [Long He](https://github.com/sh190128)
- [Wenqing Wang](https://github.com/ericaaaaaaaa)
