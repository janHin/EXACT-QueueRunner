# EXACT QueueRunner 
### Decentralized worker prototype to run algorithms for EXACT.

<img width="912" alt="image" src="https://user-images.githubusercontent.com/10051592/222464596-ca1574a2-f66b-413f-bf3e-094542f424b3.png">

EXACT QueueRunner is a process that will query the processing queue on an EXACT server for processing jobs that it can perform. Each QueueRunner
might server different plugins. 

Integration between the plugins and the QueueRunner is done through a dictionary object that needs to be provided as **plugin**:

```python 
plugin = {  'name':'MIDOG 2022 Mitosis Domain Adversarial Baseline',
            'author':'Frauke Wilm / Marc Aubreville', 
            'package':'science.imig.midog2022.baseline-da', 
            'contact':'marc.aubreville@thi.de', 
            'abouturl':'https://github.com/DeepPathology/EXACT-QueueRunner/', 
            'icon':'handlers/MIDOG2022/midog_2022_logo.png',
            'products':[],
            'results':[],
            'inference_func' : inference}
```

## Add a new PlugIn

For a new plugin, the following files need to be added:
- PlugIn (e.g. see [plugin-midog2022.py](QueueRunner/handlers/plugin-midog2022.py))
- Inference skript (e.g. see [MIDOG2022_inference.py](QueueRunner/handlers/MIDOG2022/MIDOG2022_inference.py))
> [!TIP]
> Create a separate folder for each PlugIn to save extra inference files/packages.

## Available PlugIns

Currently, the following PlugIns are available. 


- The MIDOG 2021 baseline multi-domain mitotic figure detector by Frauke Wilm trained with domain adversarial training. For more about the approach, please have a look at the following paper: 
```citation
Wilm, Frauke, et al. "Domain adversarial RetinaNet as a reference algorithm for the MItosis DOmain generalization challenge." Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis: MICCAI 2021 Challenges: MIDOG 2021, MOOD 2021, and Learn2Reg 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27–October 1, 2021, Proceedings. Cham: Springer International Publishing, 2022. 5-13.
```

- The MIDOG 2022 baseline multi-domain mitotic figure detector by Frauke Wilm trained with domain adversarial training and H&E augmentation. For more about the approach, please have a look at the following paper: 
```citation
Ammeling, Jonas, Wilm, Frauke, et al. "Reference Algorithms for the Mitosis Domain Generalization (MIDOG) 2022 Challenge." MICCAI Challenge on Mitosis Domain Generalization. Cham: Springer Nature Switzerland, 2022. 201-205.
```

- A pan-tumor T-lymphocyte detector for CD3-stained immunohistochemistry samples. For more about the approach, please have a look at the following paper: 
```citation
Wilm, Frauke, et al. "Pan-tumor T-lymphocyte detection using deep neural networks: Recommendations for transfer learning in immunohistochemistry." Journal of Pathology Informatics 14 (2023): 100301.
```



