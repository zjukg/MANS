# MANS

- [MANS: Modality-Aware Negative Sampling for Multi-modal Knowledge Graph Embedding](https://arxiv.org/abs/2304.11618)

> Negative sampling (NS) is widely used in knowledge graph embedding (KGE), which aims to generate negative triples to make a positive-negative contrast during training. However, existing NS methods are unsuitable when multi-modal informa- tion is considered in KGE models. They are also inefficient due to their complex design. In this paper, we propose Modality- Aware Negative Sampling (MANS) for multi-modal knowledge graph embedding (MMKGE) to address the mentioned problems. MANS could align structural and visual embeddings for entities in KGs and learn meaningful embeddings to perform better in multi-modal KGE while keeping lightweight and efficient. Empirical results on two benchmarks demonstrate that MANS outperforms existing NS methods. Meanwhile, we make further explorations about MANS to confirm its effectiveness.



## Prerequisites
### Build Environment
Our implemention is based on [OpenKE](https://github.com/thunlp/OpenKE). You should build Python dependencies based on the initialization steps in OpenKE.

### Download Visual Embeddings
The visual embeddings of FB15K and DB15K datasets can be download from the [GoogleDrive](https://drive.google.com/drive/folders/1D6uPpEYaoCIBxgiCT39d0u22UmlqNFMj?usp=sharing). To run the code, you should download the embeddings and place them in `visual/`

## Train & Eval
You can run the shell scripts placed in `scripts/` to run the negative sampling methods proposed in MANS. For example:
- MANS-V: run_visual_fb15k.sh
- MANS-T: run_twostage_fb15k.sh
- MANS-H: run_hybrid_fb15k.sh
- MANS-A: run_adaptive_fb15k.sh


Note that you can open the .sh file for parameter modification.

