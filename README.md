# Cortical morphometric clustering
#### Clustering of individuals with neuropsychiatric disorders using multi-modal cortical morphometrics  

Code to perform cortical clustering in the [LA5c Phenomics](https://doi.org/10.12688/f1000research.11964.2) cohort

Data available from: https://openneuro.org/datasets/ds000030/versions/1.0.0  

Preprocessed data downloaded (06/2019) via AWS S3 protocol via :  
s3://openneuro/ds000030/ds000030_R1.0.5/uncompressed/derivatives/




##### __Dependencies__
__Code was run using:__  
numpy 1.18  
scipy 1.4  
pandas 1.0.3  
scikit-learn 0.22  
pyyaml 5.3.1
neurocombat 0.1

All installed packages are shown in req.txt   
To clone environment try: `conda create -n new environment --file req.txt`

##### __Steps__  
__1. Decompose cortical metrics into non-negative components using NMF__  
>`python A__runNMF.py`

__2. Build classification models of HC vs ADHD/SCZ/BPD
> `python B__getModelExplanations.py`  
