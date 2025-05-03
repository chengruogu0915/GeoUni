<div align="center">
<br>
<img src="images/logo.jpg" width="166">
<h2>A Unified Model for Generating Geometry Diagrams, Problems and Problem Solutions</h2>

[Jo-Ku Cheng](https://chengruogu0915.github.io/)<sup>1&#42;</sup>&nbsp;
Zeren Zhang<sup>1&#42;</sup>&nbsp;
Ran Chen<sup>1</sup>&nbsp;
<br>
Jingyang Deng<sup>1</sup>&nbsp;
Ziran Qin<sup>2</sup>&nbsp;
[Jinwen Ma](https://www.math.pku.edu.cn/teachers/jwma/homepage/)<sup>1</sup>


<sup>1</sup> Peking University&nbsp; <sup>2</sup> Shanghai Jiao Tong University&nbsp;
 
[![ArXiv](https://img.shields.io/badge/ArXiv-<2504.10146>-<COLOR>.svg)](https://arxiv.org/abs/2504.10146) 

</div>


**Geo-MAGVIT**: 🤗 [Geo-MAGVIT](https://huggingface.co/JO-KU/Geo-MAGVIT) | **GeoUni**: 🤗 [GeoUni-Instruct](https://huggingface.co/JO-KU/GeoUni-Instruct) | **GeoUni-Reasoning-Adapter**: 🤗 [GeoUni-Reasoning-Adapter](https://huggingface.co/JO-KU/GeoUni-Reasoning-Adapter) 


<p align="center">
  <img src="images/GeoUni.png"  width="100%" height="auto">
</p>

### Quick Start
Before running the script, install the following necessary dependencies.

```shell
pip install torch==2.4.0 transformers==4.40.0 accelerate pillow sentencepiece
```

### Infer

Inference demo for **Diagram Generation**.
```python
python simple_infer.py
```

Inference demo for **Problem Solution**.
```python
python simple_infer.py
```

Inference demo for **Problem Creation**.
```python
python simple_infer.py
```

### Citation
To cite the paper and model, please use the below:

```
@misc{cheng2025geouniunifiedmodelgenerating,
      title={GeoUni: A Unified Model for Generating Geometry Diagrams, Problems and Problem Solutions}, 
      author={Jo-Ku Cheng and Zeren Zhang and Ran Chen and Jingyang Deng and Ziran Qin and Jinwen Ma},
      year={2025},
      eprint={2504.10146},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.10146}, 
}
```


  
