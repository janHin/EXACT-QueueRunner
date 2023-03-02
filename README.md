# EXACT QueueRunner 
### Decentralized worker prototype to run algorithms for EXACT.

<img width="912" alt="image" src="https://user-images.githubusercontent.com/10051592/222464596-ca1574a2-f66b-413f-bf3e-094542f424b3.png">

EXACT QueueRunner is a process that will query the processing queue on an EXACT server for processing jobs that it can perform. Each QueueRunner
might server different plugins. 

As a demo plug-in, the MIDOG 2022 baseline multi-domain mitotic figure detector by Frauke Wilm is provided.
For more about the approach, please have a look at the following paper: 

- Wilm, Frauke, et al. "Domain adversarial RetinaNet as a reference algorithm for the MItosis DOmain generalization challenge." Biomedical Image Registration, Domain Generalisation and Out-of-Distribution Analysis: MICCAI 2021 Challenges: MIDOG 2021, MOOD 2021, and Learn2Reg 2021, Held in Conjunction with MICCAI 2021, Strasbourg, France, September 27–October 1, 2021, Proceedings. Cham: Springer International Publishing, 2022. 5-13.

Integration between the plugins and the QueueRunner is done through a dictionary object that needs to be provided as **plugin**, see the file: [QueueRunner/handlers/midog2022.py](midog2022.py):

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

