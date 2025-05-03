# GeoUni: A Unified Model for Generating Geometry Diagrams, Problems and Problem Solutions

**Geo-MAGVIT**: 🤗 [Geo-MAGVIT](https://huggingface.co/JO-KU/Geo-MAGVIT) | **GeoUni**: 🤗 [GeoUni-1.5B](https://huggingface.co/JO-KU/GeoUni-Instruct) | **GeoUni-Reasoning-Adapter**: 🤗 [GeoUni-Reasoning-Adapter](https://huggingface.co/JO-KU/GeoUni-Reasoning-Adapter) 


<p align="center">
  <img src="images/pipeline.pdf" alt="Alt text" width="100%" height="auto">
</p>


## Quick Start
Before running the script, install the following necessary dependencies.

```shell
pip install torch==2.4.0 transformers==4.40.0 accelerate pillow sentencepiece
```


```python
python simple_infer.py
```


## Citation

```shell
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