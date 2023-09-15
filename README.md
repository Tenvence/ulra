<div align="center">
<h1> Aggregating Multiple Heuristic Signals as Supervision for Unsupervised Automated Essay Scoring
</center> <br> <center> </h1>

<p align="center">
Cong Wang, Zhiwei Jiang<sup>*</sup>, Yafeng Yin, Zifeng Cheng, Shiping Ge, Qing Gu
<br>
State Key Laboratory for Novel Software Technology, Nanjing University
<br>
<sup>*</sup> Corresponding author
<br><br>
ACL 2023 (long)<br>
[<a href="https://aclanthology.org/2023.acl-long.782/" target="_blank">paper</a>]
[<a href="https://tenvence.github.io/files/ulra-poster.pdf" target="_blank">poster</a>] 
[<a href="https://tenvence.github.io/files/ulra-slides.pdf" target="_blank">slides</a>]
<br>
</div>

## Abstract

Automated Essay Scoring (AES) aims to evaluate the quality score for input essays. In this work, we propose a novel unsupervised AES approach ULRA, which does not require groundtruth scores of essays for training. The core idea of our ULRA is to use multiple heuristic quality signals as the pseudo-groundtruth, and then train a neural AES model by learning from the aggregation of these quality signals. To aggregate these inconsistent quality signals into a unified supervision, we view the AES task as a ranking problem, and design a special Deep Pairwise Rank Aggregation (DPRA) loss for training. In the DPRA loss, we set a learnable confidence weight for each signal to address the conflicts among signals, and train the neural AES model in a pairwise way to disentangle the cascade effect among partial-order pairs. Experiments on eight prompts of ASPA dataset show that ULRA achieves the state-of-the-art performance compared with previous unsupervised methods in terms of both transductive and inductive settings. Further, our approach achieves comparable performance with many existing domain-adapted supervised models, showing the effectiveness of ULRA. The code is available at https://github.com/tenvence/ulra.

## Reference
```
@inproceedings{wang2023aggregating,
  title={Aggregating Multiple Heuristic Signals as Supervision for Unsupervised Automated Essay Scoring},
  author={Wang, Cong and Jiang, Zhiwei and Yin, Yafeng and Cheng, Zifeng and Ge, Shiping and Gu, Qing},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={13999--14013},
  year={2023}
}
```
