# Reasoning over Public and Private Data in Retrieval-Based Systems

Simran Arora, Patrick Lewis, Angela Fan, Jacob Kahn*, Christopher Ré*

[**Paper**](https://arxiv.org/abs/2203.11027)
| [**Blog Post**](https://ai.facebook.com/blog/building-systems-to-reason-securely-over-private-data)
| [**Download**](#getting-the-dataset)
| [**Citing**](#citation)

This repository contains dataset resources and code for ConcurrentQA, a textual QA benchmark to require concurrent retrieval over multiple data-distributions and privacy scopes. It also contains result analysis code and other resources for research in the private QA setting.

<p align="center"><img width="85%" src="imgs/main_figure.png" /></p>

### Getting the ConcurrentQA Dataset
The dataset can be downloaded with the provided script. Run:
```bash
bash scripts/download_cqa.sh
```
to download train, dev, and test sets along with dataset corpuses.

### Model Checkpoints
Models trained on HotpotQA be downloaded with the provided script. Run:
```bash
bash scripts/download_hotpot.sh
```
to download HotpotQA pretrained retriever and reader models.

Models trained on ConcurrentQA coming soon!

### Set up
Please follow the environment set up instructions in ```MDR_PAIR/README.md'''

### Code
We include instructions 1) for training models on ConcurrentQA and 2) for evaluating performance under the PAIR privacy framework.

#### Training Models on ConcurrentQA
Set options in the script on lines marked ``FILL IN'', and run the script as follows: 
```bash
cd MDR_PAIR
bash CQA_Scripts/MDR_end2end_CQA.sh
```

#### Evaluating QA Performance Under PAIR Framework
Set options in the script on lines marked ``FILL IN'', and run the script as follows: 
```bash
cd MDR_PAIR
bash CQA_Scripts/MDR_PairBaselines.sh
```

## Citation
Please use the following Bibtex when using the dataset:
```
@misc{arora2022reasoning,
      title={Reasoning over Public and Private Data in Retrieval-Based Systems}, 
      author={Simran Arora and Patrick Lewis and Angela Fan and Jacob Kahn and Christopher Ré},
      year={2022},
      eprint={2203.11027},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

## License
ConcurrentQA and related code is under an MIT license. See [LICENSE](LICENSE) for more information.
