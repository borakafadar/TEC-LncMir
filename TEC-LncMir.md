_Briefings in Bioinformatics_ , 2025, **26(1)** , bbaf046 

**https://doi.org/10.1093/bib/bbaf046 Problem Solving Protocol** 

**==> picture [65 x 52] intentionally omitted <==**

## **Introducing TEC-LncMir for prediction of lncRNA-miRNA interactions through deep learning of RNA sequences** 

Tingpeng Yang 1,2, Yonghong He1,2,*, Yu Wang1,*,‡ 

1Pengcheng Laboratory, No. 2, Xingke 1st Street, Nanshan District, Shenzhen, Guangdong Province 518055, China 

2Tsinghua Shenzhen International Graduate School, University Town, Nanshan District, Shenzhen, Guangdong Province 518055, China 

*Corresponding authors. Yonghong He, E-mail: heyh@sz.tsinghua.edu.cn; Yu Wang. E-mail: wangy20@pcl.ac.cn 

‡Yu Wang: Lead contact. 

## Abstract 

The interactions between long noncoding RNA (lncRNA) and microRNA (miRNA) play critical roles in life processes, highlighting the necessity to enhance the performance of state-of-the-art models. Here, we introduced TEC-LncMir, a novel approach for predicting lncRNA-miRNA interaction using Transformer Encoder and convolutional neural networks (CNNs). TEC-LncMir treats lncRNA and miRNA sequences as natural languages, encodes them using the Transformer Encoder, and combines representations of a pair of microRNA and lncRNA into a contact tensor (a three-dimensional array). Afterward, TEC-LncMir treats the contact tensor as a multichannel image, utilizes a four-layer CNN to extract the contact tensor’s features, and then uses these features to predict the interaction between the pair of lncRNA and miRNA. We applied a series of comparative experiments to demonstrate that TEC-LncMir significantly improves lncRNA-miRNA interaction prediction, compared with existing state-of-the-art models. We also trained TEC-LncMir utilizing a large training dataset, and as expected, TEC-LncMir achieves unprecedented performance. Moreover, we integrated miRanda into TEC-LncMir to show the secondary structures of high-confidence interactions. Finally, we utilized TEC-LncMir to identify microRNAs interacting with lncRNA NEAT1, where NEAT1 performs as a competitive endogenous RNA of the microRNAs’ targets (mRNAs) in brain cells. We also demonstrated the regulatory mechanism of NEAT1 in Alzheimer’s disease via transcriptome analysis and sequence alignment analysis. Overall, our results demonstrate the effectivity of TEC-LncMir, suggest a potential regulation of miRNAs by NEAT1 in Alzheimer’s disease, and take a significant step forward in lncRNA-miRNA interaction prediction. 

Keywords: lncRNA-miRNA interaction; transformer encoder; convolutional neural networks; NEAT1; Alzheimer’s disease; competitive endogenous RNA 

## Introduction 

Long noncoding RNAs (lncRNAs) are a class of noncoding RNAs usually longer than 200 ribonucleotides in length [1] and microRNAs (miRNAs) are a class of short noncoding RNAs consisting of ∼22 ribonucleotides [2, 3]. In recent years, an increasing number of studies have shown that lncRNA and miRNA play significant regulatory roles in gene expression [4, 5], cancer development [6], aging [7], neurodegenerative diseases [8], etc., where lncRNAmiRNA interaction is one of the key mechanisms [9, 10]. 

RNA sequencing technology [11] reveals numerous novel RNA sequences, including many lncRNAs and miRNAs. However, the potential interaction networks between these lncRNAs and miRNAs are largely unknown. Wet experimental techniques such as RNA immunoprecipitation [12] and RNA cross-linking [13] are both resource-intensive and time-consuming, making them insufficient to meet the challenges of high-throughput identification of lncRNA-miRNA interactions. Therefore, computational tools are developed to predict lncRNA-miRNA interactions. 

Initially, lncRNA-miRNA interaction prediction approaches, such as group-preference Bayesian collaborative filtering (GBCF) [14] and Expression Profile-based prediction model for LncRNAMiRNA Interactions (EPLMI) [15], rely on the expression data of lncRNA and miRNA. These approaches make predictions by 

comparing the expression similarities between unknown lncRNAmiRNA pairs and the known pairs. However, the expression data of lncRNA and miRNA are usually tissue-specific, inconsistent, and even not available most times; thus, GBCF and EPLMI show poor performance and can’t be used when expression data are not available. Nowadays, the sequence data of lncRNAs and miRNAs are easily available and more reliable, so approaches in recent years are mostly based on lncRNA and miRNA sequences. Specifically, Graph Embedding Ensemble Learning (GEEL) [16] utilizes a graph auto-encoder model to represent the lncRNA/miRNA sequence and a random forest classifier [17] to predict the potential interactions between lncRNAs and miRNAs. Another algorithm, PmliPred [18], proposes an approach involving a BiGRU [19] model and a random forest model to train a hybrid model for lncRNA-miRNA interaction prediction. Furthermore, LncMirNet [20] computes multiple embeddings of lncRNA and miRNA based on their sequences and utilizes convolutional neural networks (CNNs) [21] to predict lncRNAmiRNA interaction. LncMirNet also introduces other popular models (i.e. BiLSTM [22], Subgraphs, Embeddings and Attributes for Link prediction (SEAL) [23], Singular Value Decomposition (SVD) [24], Katz [25]) as compared methods. In addition, preMLI [26] employs rna2vec pretraining and a deep feature mining mechanism to predict lncRNA-miRNA interactions, PmliPEMG [27] 

**Received:** September 18, 2024. **Revised:** December 30, 2024. **Accepted:** January 22, 2025 © The Author(s) 2025. Published by Oxford University Press. 

This is an Open Access article distributed under the terms of the Creative Commons Attribution Non-Commercial License (https://creativecommons.org/ licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact journals.permissions@oup.com 

2 | Yang _et al._ 

utilizes multilevel information enhancement and a greedy fuzzy decision approach for lncRNA-miRNA interaction prediction, and GCNCRF [28] leverages a graph convolutional network (GCN) and a conditional random field (CRF) to predict human lncRNA-miRNA interactions. However, these approaches’ performance is limited, and lncRNA-miRNA interaction prediction still has a long way to go to meet the needs of practical applications. Specifically, the models’ overall performance is the key criterion for assessing their potential in practical applications. As reported in previous studies, some state-of-the-art models tend to improve lncRNAmiRNA interaction prediction in specific evaluation metrics but demonstrate limited performance in others. For instance, LncMirNet exhibits a high sensitivity of 91.58%, yet its specificity is only 79.10%, resulting in a significant gap of 12.48% between the two metrics, along with a low MCC of 71.24%. Furthermore, GEEL, PmliPred, and other approaches also report low MCCs, ranging from 19.30% to 64.45%. Therefore, our study aims to propose an approach with better overall performance, demonstrating superior results across all evaluation metrics. Additionally, we plan to train our model on a larger dataset (as compared to the datasets used by previous studies) to further enhance its performance in generalization. 

In this study, we treated lncRNA and miRNA sequences as sentences in natural languages and divided them into words using the _k-_ mer method [29]. Afterward, we presented an efficient and accurate model, TEC-LncMir, for predicting lncRNA-miRNA interaction by leveraging the power of the Transformer Encoder [30] and CNNs. Specifically, TEC-LncMir employs two Transformer Encoders to capture meaningful representations of lncRNA and miRNA sequences, respectively. These representations are then scaled and fused to generate a contact tensor, which is subsequently fed into CNNs for feature extraction. Finally, TEC-LncMir predicts the potential lncRNA-miRNA interaction based on the extracted features. Through a comprehensive series of comparative experiments with state-of-the-art models, we demonstrated that TEC-LncMir achieves significant performance improvements in lncRNA-miRNA interaction prediction. Moreover, we trained a powerful TEC-LncMir using a larger dataset, and TEC-LncMir shows superior performance. 

Nuclear Paraspeckle Assembly Transcript 1 (NEAT1) is a long noncoding RNA with a length of 23 kb [31]. Previous studies have reported that NEAT1 functions as a gene expression regulator in mammary gland development [32], cancers [31, 33], and autoimmune diseases [34, 35]. Furthermore, NEAT1 is upregulated in neurodegenerative diseases, such as Alzheimer’s disease [36]. However, the regulatory mechanism of NEAT1 in the progression of neurodegenerative diseases remains unclear. In this study, we also utilized the powerful TEC-LncMir trained on the larger dataset to predict miRNAs that interact with lncRNA NEAT1 and the results may suggest that NEAT1 is a competitive endogenous RNA (ceRNA) [37] of miRNA targets (corresponding mRNAs). Consequently, we revealed a potential regulatory role of NEAT1 in Alzheimer’s disease through transcriptome analysis and sequence alignment analysis. These results show that TECLncMir performs well in practical applications and provides new perspectives for lncRNA-miRNA interaction prediction. 

## Materials and methods 

## **Datasets** 

## _LncRNA-miRNA interaction datasets_ 

We constructed two datasets based on the lncRNASNP2 [38] database that is available at http://bioinfo.life.hust.edu.cn/static/ 

lncRNASNP2/downloads/miRNA_lncRNA_experiment, and we defined them as the base dataset and the large dataset. The base dataset is the dataset used in previous studies, so we utilized it to train TEC-LncMir and fairly compare TEC-LncMir’s performance with existing state-of-the-art models. The large dataset is ∼2.7 times larger than the base dataset and is utilized to train a more powerful TEC-LncMir model. 

The lncRNASNP2 database gathers all the lncRNA-miRNA interactions validated by wet experiments. For the base dataset, we utilized the same screening strategy used in the previous studies. Specifically, only lncRNA–miRNA interactions where the lncRNA name (Ensemble ID) starts with “ENST” and the miRNA name starts with “hsa-miR” are kept as positive pairs in the base database. As a result, 15 386 lncRNA-miRNA interactions are obtained. Furthermore, the Knuth–Durstenfeld shuffle algorithm [39] is used to shuffle the lnRNAs and miRNAs involved in the positive pairs. Afterward, an lncRNA and an miRNA are randomly sampled from the lncRNAs and miRNAs individually, and the lncRNA-miRNA pair is defined as a negative pair for the base dataset if it is not in the positive pairs. For the balance of positive and negative pairs, the sampling process is repeated until 15 386 negative pairs are obtained. Notably, the lncRNA sequences are annotated with the GENCODE [40] database (the Encyclopedia of Genes and Gene Variants, https://ftp.ebi.ac.uk/ pub/databases/gencode/Gencode_human/release_33/gencode. v33.lncRNA_transcripts.fa.gz) and the miRNA sequences are annotated with the miRbase database [41] (https://mirbase. org/download/mature.fa). For the large dataset, we kept all the experimentally validated lncRNA-miRNA interactions unless the lncRNA sequence involved is longer than 25 kb (due to Graphic Process Unit, GPU memory constraints). We also utilized the same sampling process to construct the negative pairs, and, as a result, 41 356 positive pairs and 41 356 negative pairs were obtained for the large dataset. Here, the lncRNA sequences are annotated with the NONCODE [42] database (an integrated knowledge database dedicated to non-coding RNAs, http://www.noncode. org/datadownload/NONCODEv6_human.fa.gz), and the miRNA sequences are annotated with the miRbase database. 

## _Gene expression dataset for Alzheimer’s disease_ 

We applied a series of analyses (i.e. differential gene expression analysis, correlation analysis, GO and KEGG enrichments) on the gene expression omnibus (GEO) [43] dataset GSE5281 to verify the performance of TEC-LncMir in predicting lncRNA-miRNA interactions. The GSE5281 dataset collected 13 normal brain samples and 10 Alzheimer’s disease (AD) brain samples to perform Affymetrix U133 Plus 2.0 array analysis [44]. For each sample, layer III pyramidal cells in the white matter of six brain regions (entorhinal cortex, hippocampus, medial temporal gyrus, posterior cingulate, superior frontal gyrus, and primary visual cortex) were used for total RNA extraction and array analysis. Correspondingly, the gene expression data of normal and AD samples in six brain regions were obtained. 

## **The architecture of TEC-LncMir** 

TEC-LncMir is a deep learning model consisting of five key components: lncRNA and miRNA encoders, lncRNA and miRNA scalers, an lncRNA-miRNA contact module, a contact feature extraction module, and an lncRNA-miRNA interaction prediction module. The lncRNA encoder and miRNA encoder each utilize a _k_ -mer encoder, a positional encoder, and a Transformer Encoder to encode their respective RNA sequences, yielding embeddings for both lncRNAs and miRNAs. Subsequently, the embeddings 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 3 

Figure 1. The architecture of TEC-LncMir and the data augmentation strategy. 

undergo dimension reduction using the lncRNA and miRNA scalers, and these reduced representations are integrated into the contact tensor within the lncRNA-miRNA contact module. Finally, CNNs are employed to extract features from the contact tensor, and lncRNA-miRNA interaction predictions are made based on these features. Refer to Fig. 1A for a graphical representation of TEC-LncMir. 

## _lncRNA encoder_ 

As shown in Fig. 1A, the lncRNA encoder divides a long lncRNA sequence of length _m_ into _k_ -mer segments. Specifically, a sliding window with a length of _k_ moves in steps of _k_ (using a nonoverlapping reading method) from the 5[′] end to the 3[′] end of the lncRNA sequence. The final segment, which might have a length less than _k_ , is then discarded. Consequently, the lncRNA sequence is represented by these _l_ segments ( _l_ = _m//k_ , _//_ represents the integer division operator). Considering all possible _N_ segments ( _N_ = 4 _[k]_ ) along with the zero-padding symbol [30], we constructed the “vocabulary” for lncRNAs. The _k_ -mer encoder (an embedding layer [45]) is then employed to convert the lncRNA ( _l_ segments) into a tensor of size _l_ × _d_ 0 (denoted as _X[lncRNA]_ ∈ _R[l]_[×] _[d]_[0] ). Additionally, the positional encoder [30] encodes the location information of the segments as _P[lncRNA] X_ ∈ _R[l]_[×] _[d]_[0] , and the vector of the _ith_ segment is calculated as follows: 

the embeddings of lncRNA ( _E[lncRNA]_ ∈ _R[l]_[×] _[d]_[0] ) into representations as _T[lncRNA]_ ∈ _R[l]_[×] _[d]_[1] , calculated as: 

**==> picture [167 x 11] intentionally omitted <==**

where _W_ ∈ _R[d]_[0][×] _[d]_[1] and _b_ ∈ _R[d]_[1] represent the learned weights and biases of the linear layer, respectively. The _ReLU_ function, also known as the rectified linear unit, is a nonlinear activation function defined as _ReLU(x)_ = max _(_ 0, _x)_ . 

## _miRNA encoder_ 

The structure of the miRNA encoder closely resembles that of the lncRNA encoder. The distinction lies in their respective _k_ parameters for _k_ -mer processing. Specifically, the miRNA encoder utilizes the 1-mer method, applying a 1-mer encoder to encode the individual bases (1-mer segments) due to the short length of miRNA sequences. In this context, 1-mer represents a single base, and the 1-mer encoder is also referred to as the base encoder. Consequently, the “vocabulary” for miRNAs comprises the four RNA bases and the zero-padding symbol (0: zero-padding symbol, 1: A, 2: G, 3: C, 4: U). As a result, an miRNA is encoded as _E[miRNA]_ ( _R[n]_[×] _[d]_[0] ) and transformed into representations as _T[miRNA]_ ∈ _R[n]_[×] _[d]_[1] . 

## _The lncRNA–miRNA contact module_ 

**==> picture [126 x 35] intentionally omitted <==**

where _i_ = 1, 2, · · · , _l_ , _j_ = 1, 2, · · · , _d_ 0 _/_ 2. Afterward, _X[lncRNA]_ and _P[lncRNA] X_ are added together and fed into the Transformer Encoder to generate embeddings of the lncRNA ( _E[lncRNA]_ ∈ _R[l]_[×] _[d]_[0] ). Notably, the Transformer Encoder is composed of _nl_ encoder layers with _nh_ attention heads. The dimension of the Transformer Encoder is _d_ 0, the feedforward module has a dimension of 2 _d_ 0, and the dropout parameter is set _pdropout_ (dropout means randomly setting _pdropout pdropout_ ∈ [0, 1] of values to be zero during training) [46]. Finally, the lncRNA scaler, composed of a linear layer [47], an activation function _ReLU_ [48], and a dropout layer [46], transforms 

The lncRNA-miRNA contact module fuses the representations of lncRNA and miRNA by constructing two kinds of features, computed as: 

**==> picture [90 x 43] intentionally omitted <==**

where _i_ = 1, · · · , _l_ , _j_ = 1, · · · , _n_ , ⊖ indicates the element-wise difference and indicates the Hadamard product. Consequently, _diffi_ , _j_ , _muli_ , _j_ ∈ _R[d]_[1] and _diff_ , _mul_ ∈ _R[l]_[×] _[n]_[×] _[d]_[1] . Finally, the _contact_  tensor_ is calculated as the concatenation of _diff_ and _mul_ , resulting in the _contact_  tensor_ ∈ _R[n]_[×] _[l]_[×][2] _[d]_[1] . 

4 | Yang _et al._ 

Table 1. The hyperparameters of four convolutional layers. 

|||**in_channels**|**out_channels**|**kernel_size**|**Stride**|**Padding**|**Activation function**|
|---|---|---|---|---|---|---|---|
|Layer|1|2_d_1|_d_1|_ks_|1|_ks//_2|_ReLU(x)_|
|Layer|2|_d_1|_d_1_/_2|_ks_|1|_ks//_2|_ReLU(x)_|
|Layer|3|_d_1_/_2|_d_1_/_4|_ks_|1|_ks//_2|_ReLU(x)_|
|Layer|4|_d_1_/_4|1|_ks_|1|_ks//_2|_σ(x)_|



## _The convolutional neural network module_ 

We utilized four-layer CNNs to extract the contact tensor’s features. Specifically, each convolutional layer is accompanied by a batch normalization layer [49] and a nonlinear activation function. The hyperparameters for the convolutional layers are shown in Table 1. 

_σ(x)_ is also a nonlinear activation function [50], calculated as: 

**==> picture [53 x 18] intentionally omitted <==**

The contact map ( _p_  map_ ∈ _R[l]_[×] _[n]_ ) is obtained after processing the contact tensor with the CNN module. 

## _The probability calculation module_ 

In this module, a global pooling operation is applied to _p_  map_ , which is calculated as 

**==> picture [203 x 15] intentionally omitted <==**

**==> picture [174 x 87] intentionally omitted <==**

**==> picture [81 x 34] intentionally omitted <==**

_mean_ � _p_  map_ �, _var_ � _p_  map_ � represent the mean and variance of the _p_  map_ , respectively. Furthermore, _γ_ and _η_ are learned parameters. 

## **TEC-LncMir training** 

TEC-LncMir has ∼1.12 million parameters as _nh_ , _nl_ , _ks_ , _d_ 0, _d_ 1, and _k_ are specified as 1, 4, 1, 128, 64, and 4, through a series of hyperparameter adjustment experiments. The primary training objective is to minimize the binary cross entropy (BCE) [51] loss between the predicted probabilities outputted by TEC-LncMir and the true binary labels. The model training is conducted using Python 3.8 and PyTorch 1.13.1 on an NVIDIA Tesla V100 with 32GB of memory. Moreover, the model weights are initialized with a random seed of 1234 while the training is performed for 300 epochs using a batch size of 16 and the Adam optimizer [52] with a learning rate of 0.0001. In addition, _pdropout_ is set to be 0 because we didn’t observe the over-fitting during the training. 

## **Data augmentation** 

We employed an optional data augmentation strategy to augment the number of lncRNA-miRNA pairs, thereby enhancing TECLncMir’s performance. Specifically, for a given lncRNA-miRNA pair, we divided the lncRNA sequence into _k_ -mer segments, starting from the _ith_ base, where _i_ ranges from 1 to _k_ − 1. This resulted in extending the lncRNA-miRNA pair into _k_ pairs. Figure 1B illustrates the data augmentation with _k_ set to 4. 

## **Evaluation metrics** 

Accuracy, sensitivity, specificity, positive predictive value (PPV), negative predictive value (NPV), F1 score, and the Matthews correlation coefficient (MCC) are common metrics used for evaluating binary classification problems [53]. Given the labels and predictions of lncRNA-miRNA pairs, the true positive (TP), false positive (FP), true negative (TN), and false negative (FN) samples are labeled, and the evaluation metrics are calculated as follows: 

**==> picture [177 x 161] intentionally omitted <==**

AUC and AUPR are additional commonly used metrics in binary classification tasks. AUC represents the area under the receiver operating characteristic (ROC) curve, whereas AUPR denotes the area under the precision–recall curve [53]. 

## **Bioinformatics analyses for NEAT1-miRNA interactions** _Differential gene expression analysis_ 

We applied Student’s _t_ -test [54] to identify differentially expressed genes (DEGs) between two sample groups (i.e. AD and normal groups), and the genes that satisfy the condition _P_ -value _<_ .05 during the _t_ -test are defined as DEGs. 

## _Gene Ontology and Kyoto Encyclopedia of Genes and Genomes enrichments_ 

We utilized GO (Gene Ontology) [55] and KEGG (Kyoto Encyclopedia of Genes and Genomes) [56] enrichment analyses to identify the biological functions and pathways associated with protein-coding genes that are targeted by miRNAs interacting 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 5 

with NEAT1. Moreover, the significance of each function or pathway was assessed based on the gene ratios and _P_ -values obtained from the analytical results. As a result, we unveiled the function and regulatory mechanism of NEAT1 in Alzheimer’s disease. 

## _Correlation analysis_ 

The Pearson’s correlation coefficient [57] between an upstream gene and a downstream gene was computed to illustrate the potential gene regulation mode. A positive coefficient indicates positive regulation and vice versa. 

## _Sequence alignment analysis_ 

We utilized TargetScan 8.0 [58] to search the target genes of miRNAs and applied miRanda [59] to predict the potential binding sites of miRNAs in lncRNAs. The analysis considered five types of alignments: offset 6mer, 6mer, 7mer-A1, 7mer-m8, and 8mer. Additionally, G-U mismatches were allowed in the analysis. 

## Results 

## **Hyperparameter optimization experiments for TEC-LncMir** 

The TEC-LncMir hyperparameters were determined using a series of optimization experiments on 1-fold of the base dataset. As shown in Fig. 2A, we first trained TEC-LncMir using different values of learning rate ( _lr_ ) and chose the best one ( _lr_ = 0.0001) where TEC-LncMir has the fastest convergence speed. In the same way, the number of attention heads ( _nh_ ) in Transformer Encoder layers and the batch size during training are specified as 1 (Fig. 2B) and 16 (Fig. 2C), respectively. Subsequently, we utilized different values of Transformer Encoder layers ( _nl_ ) and trained TEC-LncMir for 300 epochs to ensure that TEC-LncMir has completely converged. As shown in Fig. 2D, we evaluated TEC-LncMir’s performance using plenty of evaluation metrics on the validation dataset of the fold and utilized the MCC as the metric to select the bestperforming model because the MCC shows a more comprehensive evaluation. Consequently, we specified _nl_ as 4, and, using the same strategy, the kernel size of CNN ( _ks_ ), the dimension of Transformer Encoders ( _d_ 0), the dimension of scalers ( _d_ 1), and the _k_ value for _k_ -mer of the lncRNA encoder are determined to be 1 (Fig. 2E), 128 (Fig. 2F), 64 (Fig. 2G), and 4 (Fig. 2H), respectively. 

We also conducted a comparative experiment on the miRNA Encoder, comparing the performance of TEC-LncMir using _1-mer_ , _2-mer_ , _3-mer_ , and _4-mer_ methods. As shown in Supplementary Fig. S1 available online at http://bib.oxfordjournals.org/, TECLncMir demonstrates the best performance with the _1-mer_ method. Consequently, we selected the _1-mer_ method as the final approach for the miRNA Encoder in TEC-LncMir. 

## **TEC-LncMir achieves better performance using the data augmentation strategy** 

We performed a 5-fold cross-validation method on the base dataset and utilized the data augmentation strategy we proposed to improve TEC-LncMir’s performance. We speculated that the data augmentation makes the model learn more information about lncRNA sequences from different perspectives, and thus, the model shows more excellent performance. As shown in Fig. 3A, TEC-LncMir achieves the following improvements by using the data augmentation strategy (TEC-LncMir-Augmented versus TEC-LncMir-Unaugmented): accuracy (+1.73%), sensitivity (+2.38%), specificity (+1.06%), PPV (+1.19%), NPV (+2.31%), F1 score (+1.78%), MCC (+3.96%), AUC (+1.65%), and AUPR (+1.76%). 

These results prove the effectiveness of the data augmentation strategy. 

## **Ablation experiments for TEC-LncMir** 

We performed the following experiments to demonstrate the necessity of the modules of TEC-LncMir’s encoders. As shown in Fig. 1, the encoder part of TEC-LncMir is composed of four sequential components: the _k-mer_ encoder (I), the positional encoder (II), the Transformer encoder (III), and the scaler (IV). To assess the effectiveness of these components, ablation experiments were devised, considering that component I forms the fundamental module of the encoder part. Specifically, these experiments were structured as follows: (1) I + II + III + IV, (2) I + III + IV, (3) I + II + IV, and (4) I + II + III. We trained TEC-LncMir using the same training strategy and evaluated the performance with each of the four different encoders serving as the encoder part of TEC-LncMir. As shown in Supplementary Fig. S2 available online at http:// bib.oxfordjournals.org/, removing the positional encoder (II) or the Transformer encoder (III) from the encoder part results in a degradation of TEC-miTarget’s performance (Experiments 2 and 3). Specifically, the MCC of TEC-LncMir decreases from 78.47% ± 0.82% to 34.88% ± 6.89% when the positional encoder is removed. Similarly, TEC-LncMir ultimately converges to a higher loss and the MCC of TEC-LncMir drops from 78.47% ± 0.82% to 37.05% ± 6.82% when the Transformer encoder is removed. These results can be explained by analyzing the functions of the positional encoder and the Transformer encoder. The positional encoder integrates positional information into the embeddings of RNA sequences, enabling TEC-LncMir to comprehend the order or position of bases within RNA sequences. Meanwhile, the Transformer encoder captures dependencies between RNA bases and generates rich contextualized representations for RNA sequences. Additionally, although the removal of the scaler (IV) results in a nonsignificant performance decrease (Experiment 4), with the MCC dropping from 78.47% ± 0.82% to 77.38% ± 0.96%, the scaler module reduces the channel size of the _contacttensor_ from 2 _d_ 0 to 2 _d_ 1 (where _d_ 0 = 128 and _d_ 1 = 64). This reduction decreases the number of parameters in the CNN module and accelerates the running speed of TEC-LncMir. To demonstrate this, we measured the running time of TEC-LncMir during 5-fold cross-validation when predicting lncRNA-miRNA interactions. Specifically, with the scaler, the average running time of TEC-LncMir is 9.47 ± 0.34 min, while, without the scaler, it is 10.89 ± 0.39 min. This indicates a significant reduction in time cost (a relative decrease of 13.07%) when the scaler is used. Furthermore, we calculated the number of parameters in TECLncMir, revealing that the CNN module without the scaler has 3.94 times more parameters than with the scaler (43 715 versus 11 107), while TEC-LncMir without the scaler has 1.04 times more parameters than with the scaler (1 170 117 versus 1 120 997). These results demonstrate the effectiveness of the scaler module in reducing TEC-LncMir’s time cost. 

Overall, these results demonstrate the effectiveness of the encoder part, underscoring the pivotal roles played by the positional encoder, the Transformer encoder, and the scaler in TECLncMir. 

## **The evaluation of TEC-LncMir’s generalizability** 

We first evaluated the impact of the ratio of positive to negative pairs ( _r_ ) on the performance of TEC-LncMir. Specifically, we set the r values at 2:1, 1:1, and 1:2. As shown in Supplementary Fig. S3 available online at http://bib.oxfordjournals.org/, TEC-LncMir exhibits stable performance, with accuracy, sensitivity, PPV, and F1 

6 | Yang _et al._ 

Figure 2. Hyperparameter optimization experiments for TEC-LncMir. 

score remaining very similar across different ratios. Furthermore, the results indicate that specificity and NPV values increase as the ratio decreases, leading to an increase in MCC. This is reasonable because more negative pairs are learned as the ratio decreases. Moreover, we evaluated the generalizability of TEC-LncMir. Specifically, we employed a nonreplacement 

sampling method across the entire dataset and recorded the sets of lncRNA and miRNA from the extracted samples. During each sampling, we ensured that the lncRNA and miRNA in the newly extracted samples were not included in the corresponding sets of previously extracted lncRNA and miRNA. Once this sampling process was completed, we used the extracted samples 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 7 

Figure 3. Performance comparison of TEC-LncMir and the state-of-the-art models. (A) The 5-fold cross-validation results of TEC-LncMir and comparative models on the base dataset. PmliPred, preMLI, and PmliPEMG are methods with available source code; hence, we ran these models on the base dataset, and the evaluation metrics are presented as mean ± SD. For the other models, which are not open-source tools, we directly referenced their reported performance metrics (shown as mean) from the respective studies conducted on the same dataset. TEC-LncMir-Unaugmented and TECLncMir-Augmented represent TECLncMir’s performance without and with the data augmentation strategy, respectively. (B) The comparative results of TEC-LncMir versus PmliPred, preMLI, and PmliPEMG on the large dataset. (C, D) The radar charts illustrating the comparative results on the base dataset (C) and the large dataset (D). (F) The distribution map of predictions for TEC-LncMir. 

as the independent test set (10%), further dividing the remaining samples into training (80%) and validation sets (10%). Consequently, the independent test set does not share any lncRNAs and miRNAs with the training and validation sets. Then, we calculated the sequence similarity between the independent test set and the training and validation sets. Specifically, the sequence similarities of lncRNAs and miRNAs between the independent test set and the training and validation sets were 0.3791 ± 0.1235 and 0.3824 ± 0.1209, respectively. These low sequence similarities indicate that TEC-LncMir predicts interactions based on generalization rather than memorization. Finally, we trained TEC-LncMir and compared its performance on the validation and independent test sets. As shown in Supplementary Fig. S4 available online at http://bib.oxfordjournals.org/, TEC-LncMir demonstrates similar, stable, and superior performance on both the validation and independent test sets. Furthermore, we compared the generalizability of TEC-LncMir and PmliPred, and the results show that PmliPred exhibits a larger performance gap when evaluated on the validation and independent datasets. These results demonstrate the generalizability of TEC-LncMir. 

## **The framework of TEC-LncMir can also be used for miRNA-mRNA interaction prediction** 

In this study, we focused more on the interactions between lncRNAs and miRNAs because lncRNAs typically have more complex secondary structures, whereas mRNAs have usually 

simpler secondary structures. Therefore, predicting the interactions between lncRNAs and miRNAs is a more challenging task. However, TEC-LncMir can also be used to predict miRNAmRNA interactions since both mRNAs and lncRNAs are composed of the same four types of ribonucleotides. Therefore, we first constructed an independent test set, training, and validation sets on a dataset of miRNA-mRNA interaction samples using the same procedure as the previous section (the corresponding sequence similarities of mRNAs and miRNAs between the independent test set and the training and validation sets were 0.2951 ± 0.1563 and 0.4047 ± 0.0820). Then, we trained TEC-LncMir and evaluated its performance on the validation dataset and an independent test set (Supplementary Fig. S5 available online at http://bib. oxfordjournals.org/). The results demonstrate TEC-LncMir’s stable and superior performance in predicting miRNA-mRNA interactions, achieving an MCC of 97.65% on the validation set and 96.61% on the independent test set. Notably, TEC-LncMir performs significantly better in predicting miRNA-mRNA interactions than in predicting miRNA-lncRNA interactions, highlighting the greater difficulty of predicting miRNA-lncRNA interactions. 

## **TEC-LncMir outperforms the state-of-the-art models** 

We compared TEC-LncMir’s performance with state-of-the-art models such as GEEL, PmliPred, preMLI, PmliPEMG, BiLSTM, SEAL, SVD, Katz, and LncMirNet. Notably, PmliPred, preMLI and 

8 | Yang _et al._ 

PmliPEMG are methods with available source code, while the other models were not open-source tools. Consequently, we ran PmliPred, preMLI and PmliPEMG on the base dataset and directly referenced the performance results of other models as reported in their respective studies on the same dataset. As shown in Fig. 3A, TEC-LncMir, both without (TEC-LncMir-Unaugmented) and with (TEC-LncMir-Augmented) data augmentation, demonstrates better performance in the experiments. Specifically, TEC-LncMir-Unaugmented and TEC-LncMir-Augmented achieve improvements of 10.15% and 14.50% in the MCC, respectively, as compared to the best-performing state-of-the-art models. Furthermore, TEC-LncMir-Unaugmented also shows improvements in other evaluation metrics, including specificity (+5.31%), F1 score (+3.60%), accuracy (+4.56%), and AUC (+0.07%). TECLncMir-Augmented, on the other hand, demonstrates enhancements in sensitivity (+0.60%), specificity (+6.42%), F1 score (+5.44%), accuracy (+6.36%), and AUC (+1.74%). 

Afterward, we run the comparative methods with available source codes (PmliPred, preMLI, and PmliPEMG) on the large dataset to further compare them with TEC-LncMir (Fig. 3B). The results demonstrate that TEC-LncMir outperforms the other methods in these experiments. Specifically, as compared to the best-performing comparative model, preMLI, TEC-LncMirUnaugmented and TEC-LncMir-Augmented show increases in MCC by 28.37% and 30.32%, respectively. In addition, we utilized the radar chart to give a visual display of the comparative results. Specifically, we used the above six evaluation metrics as the axes of the radar chart and drew the figure for each model. The areas of the figure present the model performances, and, as expected, TEC-LncMir shows larger area in the radar chart (Fig. 3C and D). 

We then compared TEC-LncMir’s performance with GCNCRF on the same five data folds used in GCNCRF. As shown in Supplementary Table S1 available online at http://bib. oxfordjournals.org/, TEC-LncMir significantly outperforms GCNCRF in PPV (87.09% ± 4.74% versus 7.79%), specificity (99.71% ± 0.11% versus 92.54%), and AUPR (68.89% ± 8.15% versus 28.15%). Furthermore, TEC-LncMir and GCNCRF show similar performance in AUC (92.20% ± 2.56% versus 94.70%) and accuracy (98.15% ± 0.33% versus 98.14%). Notably, although GCNCRF achieves a higher sensitivity (87.95%) than TEC-LncMir (54.78% ± 9.32%), the 33.17% increase in sensitivity is outweighed by the substantial decreases in PPV (79.30%) and AUPR (40.74%). Consequently, TEC-LncMir achieves a higher F1 score (66.91% ± 7.70%) compared to GCNCRF (14.31%), demonstrating its superior overall performance. 

We also directly utilized the seed-match method miRanda to predict lncRNA-miRNA interactions on the five test sets displayed in Fig. 3 and compared its performance with TECLncMir. The results (Supplementary Fig. S6 available online at http://bib.oxfordjournals.org/) indicate that the seed-match method miRanda achieves an average MCC of only 30.83%, significantly lower than TEC-LncMir’s MCC of 81.57%. This discrepancy is expected since miRanda treats lncRNAs as linear molecules, ignoring their secondary structures. As shown in Supplementary Fig. S7 available online at http://bib.oxfordjourn als.org/, miRanda predicts an 8mer secondary structure for the interaction between the lncRNA and miRNA hsa-miR-6741-5p. However, the “target sites” in the lncRNA may interact with the bases within the lncRNA itself, leading to a secondary structure of the interaction with low confidence. Consequently, miRanda struggles to identify these “false-positive” interactions, resulting in low sensitivity and F1 score. Therefore, seed-match methods are more suitable for predicting the target sites of miRNAs in 

mRNAs, as mRNAs are usually linear molecules, whereas lncRNAs often possess secondary structures. 

## **Training powerful TEC-LncMir using the large dataset** 

Deep learning models are data-driven, so we thought that training TEC-LncMir on the large dataset would make TEC-LncMir more powerful. We also applied the 5-fold cross-validation strategy on the large dataset and compared the results with those obtained from the base dataset. As shown in Fig. 3B, TEC-LncMirUnaugmented trained on the large dataset outperforms TECLncMir-Unaugmented trained on the base dataset with the following improvements: sensitivity (+4.61%), specificity (+5.34%), F1 score (+4.93%), accuracy (+4.96%), AUC (+4.07%), and MCC (+11.29%). Furthermore, compared to TEC-LncMir-Augmented trained on the base dataset, TEC-LncMir-Augmented trained on the large dataset shows improvements in sensitivity (+4.24%), specificity (+3.52%), F1 score (+3.86%), accuracy (+3.89%), AUC (+3.05%), and MCC (+8.69%). Moreover, the distribution map of predictions (Fig. 3E) show TEC-LncMir’s accurate identification of lncRNA-miRNA interactions. These results prove that the large database brings the potential for practical applications to TEC-LncMir. Notably, the baseline methods also demonstrate significant improvements when trained on the large dataset. Specifically, PmliPred, preMLI, and PmliPEMG show increases in MCC by 5.82%, 16.17%, and 10.91%, respectively. These findings underscore that training on larger datasets improves the effectiveness of deep learning models. 

## **TEC-LncMir offers insights into the NEAT1-miRNA interactions in Alzheimer’s disease** 

The competitive endogenous RNAs (ceRNAs) are a class of noncoding RNAs that bind to miRNAs and serve as the competitors of the mRNAs regulated by the miRNAs (Fig. 4A), so ceRNAs downregulate the corresponding miRNAs naturally. Moreover, given that miRNAs bind to specific mRNAs and guide the degradation of them, miRNAs usually down-regulate the expression of mRNAs, leading to a reduction in corresponding protein synthesis. Therefore, the expressions of the ceRNA and the mRNA in a ceRNAmiRNA-mRNA network are positively correlated. Here, we pay attention to the regulatory mechanism of the lncRNA NEAT1 that is related to Alzheimer’s disease. Specifically, we aim to find the miRNAs that may interact with NEAT1 using TEC-LncMir and construct the NEAT1-miRNA-mRNA network where NEAT1 might serve as a ceRNA. 

We first applied analyses on the GEO dataset GSE5281 to propose the potential regulatory mechanism of NEAT1 in Alzheimer’s disease. As shown in Fig. 4B, NEAT1 is significantly up-regulated in six brain regions of AD samples. It is widely acknowledged that the hippocampus is the main lesion site in AD; therefore, the gene expression profiles (24,442 genes) of 23 samples from the hippocampus (AD: _n_ = 10, normal _n_ = 13) were extracted for differential gene expression analysis. As a result, a total of 6495 DEGs (3201 up-regulated and 3294 down-regulated) were screened between the AD and the normal groups with a threshold _P_ -value _<_ .05 (Fig. 4C). Among the DEGs, we obtained 24 up-regulated and 28 down-regulated miRNA genes. 

Given that NEAT1 is up-regulated in AD samples and considering the negative correlation between NEAT1 and the miRNAs interacting with it in the NEAT1-miRNA-mRNA network, we utilized TEC-LncMir to predict interactions between NEAT1 and the down-regulated miRNA genes to obtain more reliable results. 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 9 

Figure 4. The prediction and validation of NEAT1-miRNA interactions. (A) The ceRNA mechanism. (B) NEAT1 is up-regulated in brain samples of Alzheimer’s disease, six different brain regions are analyzed, and data are shown in as median with interquartile range,[∗] _P <_ .05,[∗∗] _P <_ .01,[∗∗∗] _P <_ .001, ∗∗∗∗ _P <_ .0001, Student _t_ -test. (C) Volcano map of the genes in the hippocampus region, down-regulated genes, up-regulated genes, and genes without differential expression are respectively displayed (the Alzheimer’s disease group versus the normal group). (D) The miRNA genes interacted with NEAT1, predicted by TEC-LncMir-Large. (E) The predicted five miRNA genes are significantly down-regulated in the AD hippocampus, and data are presented in the same way as in (B). (F) The expression analysis of the predicted five miRNA genes in six brain regions, data are shown as the log2 FC, and FC is calculated as FC = ME(AD)/ME(Normal), where ME indicates the mean expression. (G) The correlation analysis between NEAT1 and the predicted five miRNA genes in six brain regions. 

For each miRNA gene, two kinds of mature miRNA sequences (3p and 5p) are considered if they exist, and the miRNA gene is thought to interact with NEAT1 if any mature miRNA sequence of it has an interaction with NEAT1. Therefore, 44 NEAT1-miRNA pairs were obtained and the interaction probabilities were calculated utilizing the five TEC-LncMir-Large models generated by the 5-fold cross-validation experiments. To improve the confidence of the predictions, we used the intersection of the five models’ predictions as the final results (Fig. 4D). 

Since TEC-LncMir provides the interaction probabilities between lncRNAs and miRNAs, a threshold must be set to select interactions with high confidence. In this study, we selected the top five miRNAs with the highest confidence scores to construct the NEAT1-miRNA interaction network. The corresponding threshold for these interactions was 0.94. Consequently, five highconfidence miRNA genes (MIR1292, MIR6741, MIR7, MIR9, and MIR4738) were identified. These miRNA genes are significantly down-regulated in the hippocampus of AD brain samples (Fig. 4E). 

We further analyzed the expressions of these miRNA genes in the other five brain regions of AD samples, and, as shown in Fig. 4F, the down-regulation of these miRNA genes is universal. Moreover, we calculated the Pearson’s correlation coefficients between these miRNA genes and NEAT1, and it shows that these miRNA genes have strong negative correlations with NEAT1 in six brain regions (Fig. 4G). It is crucial to notice the negative correlation between the expression of NEAT1 and these miRNA genes, while all these miRNAs have reliable target sites on NEAT1. It did not escape our attention that by acting as miRNA target mimicry, the upregulation of NEAT1 offers a mechanism to damp the translational repression effect of miRNAs. 

## **Constructing NEAT1-miRNA-mRNA regulatory networks** 

We analyzed the target protein-coding genes of the five miRNA genes and constructed the whole NEAT1-miRNA-mRNA regulatory networks. Specifically, we employed TargetScan 8.0 

10 | Yang _et al._ 

to identify the target genes of all mature miRNAs associated with each miRNA gene. Subsequently, we selected the top 20 target genes with high confidence levels to show the networks. Notably, the confidence level was calculated as the absolute value of the total context++ score predicted by TargetScan 8.0. Afterward, we displayed the NEAT1-miRNA-mRNA regulatory networks using the software Cytoscape (Fig. 5A). We also applied function analyses utilizing GO and KEGG enrichments to reveal NEAT1’s functions in AD. As shown in Fig. 5B, these downstream protein-coding genes are primarily related to the DNA-binding transcription factor binding, RNA polymerase II-specific DNAbinding transcription factor binding, and the postsynaptic specialization. These GO terms are highly correlated with the development of Alzheimer’s disease. For example, DNA-binding transcription factor binding has been shown to control the expression of the amyloid precursor protein (APP), leading to one of the primary features of Alzheimer’s disease: the deposition of fibrillar amyloid within senile plaques in certain brain areas [60]. Furthermore, the hyperphosphorylation of RNA polymerase II (RNAP II) is reported to precede tau phosphorylation and neurofibrillary tangle formation in Alzheimer’s disease [61]. Additionally, postsynaptic specialization contributes to structural plasticity changes in the postsynaptic density (PSD), leading to a loss of molecular homeostasis within the synapse and contributing to early symptoms of Alzheimer’s disease [62]. The KEGG results (Fig. 5C) also demonstrate that these genes are involved in the FoxO signaling pathway, where FoxO activation is one of the key mechanisms underlying A _β_ -induced brain cell death in Alzheimer’s disease [63]. Moreover, these genes are also implicated in other diseases, such as glioma and various cancers. These findings support the potential mechanisms we proposed. Based on these findings, we proposed a potential regulatory mechanism of NEAT1 in Alzheimer’s disease, i.e. NEAT1 might serve as a ceRNA in the brain, and the up-regulation of NEAT1 is negatively correlated with the down-regulating the five important miRNA genes: MIR1292, MIR6741, MIR7, MIR9, and MIR4738. As a consequence, the downstream protein-coding genes are upregulated. The misregulation of the proteins induces the disorder of many signal pathways and may finally promote the occurrence and development of AD. 

We applied the correlation analysis to further confirm the NEAT1-miRNA-mRNA networks. As shown in Fig. 6A, NEAT1 has strong positive correlations with the target protein-coding genes of MIR1292, and MIR1292 exerts a negative regulatory influence on the expression of its target genes. Moreover, given that NEAT1 is highly correlated with MIR1292 (Fig. 4G), we demonstrated a potential NEAT1-MIR1292-mRNA network. Using the same strategy, we also showed highly possible interaction networks of demonstrated NEAT1-MIR6741mRNA (Supplementary Fig. S8 available online at http://bib. oxfordjournals.org/), NEAT1-MIR7-mRNA (Supplementary Fig. S9 available online at http://bib.oxfordjournals.org/), NEAT1- MIR9mRNA (Supplementary Fig. S10 available online at http://bib. oxfordjournals.org/),and NEAT1- MIR4738-mRNA (Supplementary Fig. S11 available online at http://bib.oxfordjournals.org/) networks. 

## **Sequence alignment analysis of NEAT1-miRNA interactions** 

We further utilized the sequence alignment analysis to validate the predicted NEAT1-miRNA interactions. For each mature miRNA originating from the identified miRNA genes (MIR1292, MIR6741, MIR7, MIR9, and MIR4738), we employed miRanda to identify potential binding sites within NEAT1. Notably, the 

_score_ hyperparameter in miRanda is set to its default value of 140.0, while the _energy_ hyperparameter is adjusted from 0 to −1 to enhance result reliability [64]. As shown in Fig. 7 and Supplementary Table S2 available online at http://bib. oxfordjournals.org/, NEAT1 has plenty of binding site positions, with alignments indicating a high probability of preferential conservation. Specifically, most of the alignment types are 7-merm8, and some are even 8mer. These results further demonstrate the accurate lncRNA-miRNA predictions of TEC-LncMir. 

## Discussion 

LncRNA and miRNA are important regulatory elements exerting regulatory effects at the transcriptional and post-transcriptional levels, and lncRNA-miRNA interaction is one of the most important mechanisms for these regulatory effects. Consequently, predicting lncRNA-miRNA interactions plays a crucial role in understanding the significant functions of lncRNA and miRNA in gene expression regulation, cancer development, aging, neurodegenerative diseases, etc. In recent years, more and more lncRNA and miRNA and their sequences have been revealed by RNA sequencing techniques, highlighting the need to discover novel lncRNA-miRNA interactions. However, it’s resource-intensive and time-consuming to reveal lncRNA-miRNA interactions using wet experimental techniques, highlighting the necessity to develop computational tools for predicting lncRNA-miRNA interactions. 

The rapid evolution of artificial intelligence (AI) has significantly accelerated the development of lncRNA-miRNA interaction prediction, with a particular boost from deep learning techniques. Over the past years, state-of-the-art approaches, such as GEEL, PmliPred, BiLSTM, SEAL, SVD, Katz, and LncMirNet, have gradually made lncRNA-miRNA interaction prediction more promising. However, the performance of these approaches still needs to be improved to meet the demands of practical applications. 

LncRNA sequences are usually long, so it’s difficult to encode them base by base due to GPU memory constraints. In the present study, we introduced the lncRNA encoder to represent the lncRNA sequences _k-mer_ by _k-mer_ , i.e. a sliding window with a length of _k_ moves in steps of _k_ from the 5[′] end to the 3[′] end of the lncRNA sequence and several _k-mer_ segments are obtained. The lncRNA encoder forms an integral part of our novel lncRNAmiRNA interaction prediction model, named TEC-LncMir, which is built on the Transformer Encoder and CNNs. We rigorously assess the performance of the lncRNA encoder with different values of the hyperparameter _k_ and select the most effective one as the ultimate lncRNA encoder for TEC-LncMir. Through the deep learning of ribonucleic acid sequences, TEC-LncMir obtains meaningful representations of lncRNAs and miRNAs. Afterward, TEC-LncMir fuses these representations, utilizes CNNs to extract interaction features, and makes accurate lncRNA-miRNA interaction identifications. 

We utilized a set of analyses to demonstrate the effectiveness of TEC-LncMir. Specifically, we first determined the optimal hyperparameters using a series of optimization experiments, demonstrated the effectiveness of TEC-LncMir’s modules utilizing several ablation experiments, and evaluated the impact of the ratio of positive to negative pairs on the performance of TECLncMir and TEC-LncMir’s generalizability in predicting miRNAmRNA interactions. Afterward, we applied the data augmentation strategy to improve the performance of TEC-LncMir and demonstrated that TEC-LncMir achieves significantly more accurate results than other state-of-the-art methods in a series of comparative experiments. Notably, the MCC of TEC-LncMir (81.57%) is 10.33 percentage points higher than the maximum value of 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 11 

Figure 5. NEAT1-miRNA-mRNA networks and function analyses of NEAT1. (A) The NEAT1-miRNA-mRNA networks, LPP and ZBTB20: mRNA genes regulated by several miRNAs at the same time. (B, C) GO and KEGG analyses of the protein-coding genes regulated by NEAT1. The significance threshold for GO and KEGG selection was based on gene ratios in descending order and _P_ -values _<_ .05. The top 10 items are displayed. 

Figure 6. The correlation analysis of the NEAT1-MIR1292-mRNA network. (A) NEAT1 has strong positive correlations with the target protein-coding genes of MIR1292 in six brain regions. (B) MIR1292 shows strong negative correlations with its target genes. 

12 | Yang _et al._ 

Figure 7. Sequence alignment analysis of NEAT1–miRNA interactions. (A) NEAT1-MIR1292 interaction. (B, C) NEAT1-MIR6741 interaction. (D–F) NEAT1MIR7 interaction. (G, H) NEAT1-MIR9 interaction. (I, J) NEAT1-MIR4738 interaction. 

the rest models. Furthermore, we trained TEC-LncMir on a larger dataset to improve TEC-LncMir’s performance, and, as expected, TEC-LncMir achieves a higher MCC (87.33), underscoring its practical potential. 

We also provided the curve of loss variation during the training phase and conducted a statistical analysis of the loss values across different types of test samples. As shown in Supplementary Fig. S12 available online at http://bib.oxfordjourn als.org/, the loss function of TEC-LncMir consistently decreases as the number of training epochs increases. This trend indicates an improvement in the model’s learning capability, enabling it to fit the training data more effectively. Specifically, with each epoch, the loss value exhibited a continuous downward trajectory, ultimately converging to a relatively stable level of 0.01. This convergence behavior validates the effectiveness of the optimization algorithm and hyperparameter settings employed, demonstrating that the model’s performance on the training set gradually approaches its optimal state. Furthermore, the figure illustrates a significant variation in the loss function among 

the 2830 true positives (TP), 2696 true negative (TN), 273 false positive (FP), and 355 false negative (FN) samples. Notably, the loss for TP and TN samples remains consistently low, reflecting the model’s accurate classification of these instances. In contrast, the loss for FP and FN samples is considerably higher, indicating a greater penalty for misclassifications. This disparity highlights the model’s sensitivity to errors, particularly in instances where the predictions diverge from the ground truth. 

Finally, we utilized TEC-LncMir for practical applications, i.e. we applied the powerful TEC-LncMir trained on the large dataset to predict miRNAs that have interactions with lncRNA NEAT1 and constructed the NEAT1-miRNA-mRNA regulatory networks in Alzheimer’s disease. We first analyzed the predictions of NEAT1-miRNA interactions made by the baseline method PmliPred and compared the top five miRNAs with the highest confidence scores to those predicted by TEC-LncMir. As shown in Supplementary Fig. S13A available online at http://bib. oxfordjournals.org/, MIR9 was identified as a common miRNA by both tools, while TEC-LncMir uniquely identified MIR1292, 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 13 

MIR6741, MIR7, and MIR4738. In contrast, MIR6884, MIR6758, MIR3620, and MIR6890 were identified by PmliPred but not by TECLncMir. We also assessed the correlation coefficients between the exclusive miRNAs and NEAT1 (Supplementary Fig. S13B available online at http://bib.oxfordjournals.org/), revealing that the miRNAs identified by TEC-LncMir exhibited stronger negative correlations with NEAT1 (−0.35 versus −0.24). These results indicate the greater accuracy of TEC-LncMir’s predictions. Furthermore, we analyzed bulk tissue gene expression for these miRNAs and found that they are predominantly expressed in the brain (Supplementary Fig. S14 available online at http://bib. oxfordjournals.org/), highlighting the potential impact of their downregulation on Alzheimer’s disease. Moreover, we performed the transcriptome and sequence alignment analyses to confirm the networks and applied the GO and KEGG enrichments to reveal the functions of NEAT1 in Alzheimer’s disease. The NEAT1miRNA-mRNA interaction mechanism demonstrates that the upregulation of downstream protein-coding genes is likely caused by the up-regulation of the competitive endogenous RNA NEAT1 in Alzheimer’s disease. Therefore, a potential strategy for fighting Alzheimer’s disease could involve down-regulating the expression of NEAT1, for which RNA silencing can be employed [65]. 

In this study, we divided the prediction of the secondary structure of lncRNA-miRNA interactions into two steps. First, we proposed TEC-LncMir to predict the interaction probabilities between lncRNAs and miRNAs. Next, we selected the lncRNA-miRNA pairs with high interaction probabilities and used seed-match methods (e.g. miRanda) to reveal all possible secondary structures of the interactions. We did this for the following reasons: (i) RNA structure prediction is a hot research field with many unsolved problems, and the accurate secondary structure of lncRNA-miRNA interactions remains unknown. For example, a recent study on LinearCoFold and LinearCoPartition [66] shows that these methods achieve only a PPV of 20%–30% and a sensitivity of 40%–50% in predicting secondary structures. Therefore, we can’t construct the ground-truth interactions (the secondary structure), which is a technical barrier for deep learning models. (ii) Predicting the potential secondary structures of the lncRNA-miRNA interactions is a less difficult task because the interactions between RNAs are based on seed-match methods. Therefore, popular software (e.g. miRanda) can be used to predict all possible secondary structures (e.g. 8mer, 7-mer-m8). However, these results may not be highly reliable because lncRNA is not a linear molecule and has a complex structure (Supplementary Fig. S7 available online at http:// bib.oxfordjournals.org/). To enhance reliability, we first identified high-confidence lncRNA-miRNA interactions using TEC-LncMir and then presented the secondary structures of these interactions. 

In summary, we proposed the use of TEC-LncMir with the support of deep learning methods and demonstrated that TECLncMir is accurate in lncRNA-miRNA interaction identification. The results of our study offer fresh insights into lncRNA-miRNA interaction prediction and advance it significantly toward practical applications. 

## **Key Points** 

- The model TEC-LncMir is proposed for lncRNA-miRNA interaction prediction based on Transformer Encoder and convolutional neural networks. 

- The Transformer Encoder encodes lncRNA and miRNA sequences, while the convolutional neural networks 

extract features from these representations, enabling the prediction of lncRNA-miRNA interactions. 

- TEC-LncMir outperforms state-of-the-art approaches in a series of comparative experiments. 

- TEC-LncMir performs better using a larger training dataset and performs well in identifying microRNAs that interact with lncRNA NEAT1 in Alzheimer’s disease. 

- TEC-LncMir integrates miRanda to show the secondary structures of the high-confidence interactions. 

## Acknowledgements 

The authors would like to thank Pengcheng Laboratory (PCL) for its computational support and the use of the Pengcheng Brain supercomputer. Additionally, they express their gratitude to the administrators of the Pengcheng Brain supercomputer for assisting in the training of the AI models. 

## Supplementary data 

Supplementary data are available at _Briefings in Bioinformatics_ online. 

Conflict of interest: None declared. 

## Funding 

This work has been supported by direct national funding from the Chinese Ministry of Technology to Pengcheng Laboratory, Research and Development Program of Guangzhou Laboratory (SRPG22-001), and the R&D Program of Pengcheng Laboratory, Grant No. PCL2023A09. 

## Code and data availability 

TEC-LncMir is implemented in Python, the code, the datasets used in this study, and the model weight we trained are available on GitHub (https://github.com/tingpeng17/TEC-LncMir). 

## Author contributions 

T.P.Y. conceived the study, collected the datasets, trained TECLncMir, and drafted the original manuscript. Y.W. and Y.H.H. guided the model design, analyzed the results, and edited the manuscript. All authors read and approved the final draft. 

## References 

1. Zhang X, Hong R, Chen W. _et al._ The role of long noncoding RNA in major human disease. _Bioorg Chem_ 2019; **92** :103214. https:// doi.org/10.1016/j.bioorg.2019.103214. 

2. Sass S, Dietmann S, Burk UC. _et al._ MicroRNAs coordinately regulate protein complexes. _BMC Syst Biol_ 2011; **5** :1–11. https:// doi.org/10.1186/1752-0509-5-136. 

3. Bartel DP.Metazoan microRNAs. _Cell_ 2018; **173** :20–51. https://doi. org/10.1016/j.cell.2018.03.006. 

4. Wang Z, Liao W, Liu F. _et al._ Downregulation of lncRNA EPB41L4A-AS1 mediates activation of MYD88-dependent NF- _κ_ B pathway in diabetes-related inflammation. _Diabetes Metab Syndr Obes_ 2021; **14** :265–77. https://doi.org/10.2147/DMSO.S280765. 

5. Wan G, Xie W, Liu Z. _et al._ Hypoxia-induced MIR155 is a potent autophagy inducer by targeting multiple players in the MTOR pathway. _Autophagy_ 2014; **10** :70–9. https://doi.org/10. 4161/auto.26534. 

14 | Yang _et al._ 

6. Liao M, Liao W, Xu N. _et al._ LncRNA EPB41L4A-AS1 regulates glycolysis and glutaminolysis by mediating nucleolar translocation of HDAC2. _EBioMedicine_ 2019; **41** :200–13. https://doi.org/10.1016/ j.ebiom.2019.01.035. 

7. Ghafouri-Fard S, Abak A, Talebi SF. _et al._ Role of miRNA and lncRNAs in organ fibrosis and aging. _Biomed Pharmacother_ 2021; **143** :112132. https://doi.org/10.1016/j.biopha.2021. 112132. 

8. Yang T, Wang Y, Liao W. _et al._ Down-regulation of EPB41L4A-AS1 mediated the brain aging and neurodegenerative diseases via damaging synthesis of NAD+ and ATP. _Cell Biosci_ 2021; **11** :1–14. https://doi.org/10.1186/s13578-021-00705-2. 

9. Yang G, Lu X, Yuan L. LncRNA: a link between RNA and cancer. _Biochimica et Biophysica Acta (BBA)-Gene Regulatory Mechanisms_ 2014; **1839** :1097–109. https://doi.org/10.1016/j.bbagrm.2014.08. 012. 

10. Nie L, Li C, Zhao T. _et al._ LncRNA double homeobox a pseudogene 8 (DUXAP8) facilitates the progression of neuroblastoma and activates Wnt/ _β_ -catenin pathway via microRNA-29/nucleolar protein 4 like (NOL4L) axis. _Brain Res_ 2020; **1746** :146947. https:// doi.org/10.1016/j.brainres.2020.146947. 

11. Hong M, Tao S, Zhang L. _et al._ RNA sequencing: new technologies and applications in cancer research. _J Hematol Oncol_ 2020; **13** : 1–16. https://doi.org/10.1186/s13045-020-01005-x. 

12. Gagliardi M, Matarazzo MR. Rip: Rna immunoprecipitation. _Polycomb Group Proteins: Methods and Protocols_ 2016; **1480** :73–86. https://doi.org/10.1007/978-1-4939-6380-5_7. 

13. Kudla G, Granneman S, Hahn D. _et al._ Cross-linking, ligation, and sequencing of hybrids reveals RNA–RNA interactions in yeast. _Proc Natl Acad Sci_ 2011; **108** :10010–5. https://doi.org/10.1073/ pnas.1017386108. 

14. Huang ZA, Huang YA, You ZH. _et al._ Novel link prediction for large-scale miRNA-lncRNA interaction network in a bipartite graph. _BMC Med Genet_ 2018; **11** :17–27. https://doi.org/10.1186/ s12920-018-0429-8. 

15. Huang YA, Chan KCC, You ZH. Constructing prediction models from expression profiles for large scale lncRNA–miRNA interaction profiling. _Bioinformatics_ 2018; **34** :812–9. https://doi. org/10.1093/bioinformatics/btx672. 

16. Zhou S, Yue X, Xu X. _et al._ LncRNA-miRNA interaction prediction from the heterogeneous network through graph embedding ensemble learning[C]//. In: _2019 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)_ , San Diego, CA, USA, pp. 622–7. IEEE, 2019. 

17. Rigatti SJ. Random forest. _J Insur Med_ 2017; **47** :31–9. https://doi. org/10.17849/insm-47-01-31-39.1. 

18. Kang Q, Meng J, Cui J. _et al._ PmliPred: a method based on hybrid model and fuzzy decision for plant miRNA–lncRNA interaction prediction. _Bioinformatics_ 2020; **36** :2986–92. https:// doi.org/10.1093/bioinformatics/btaa074. 

19. Zhang J, Liu F, Xu W. _et al._ Feature fusion text classification model combining CNN and BiGRU with multi-attention mechanism. _Future Internet_ 2019; **11** :237. https://doi.org/10.3390/ fi11110237. 

20. Yang S, Wang Y, Lin Y. _et al._ LncMirNet: predicting LncRNA– miRNA interaction based on deep learning of ribonucleic acid sequences. _Molecules_ 2020; **25** :4372. https://doi.org/10.3390/ molecules25194372. 

21. Gu J, Wang Z, Kuen J. _et al._ Recent advances in convolutional neural networks. _Pattern Recogn_ 2018; **77** :354–77. https:// doi.org/10.1016/j.patcog.2017.10.013. 

22. Siami-Namini S, Tavakoli N, Namin AS. The performance of LSTM and BiLSTM in forecasting time series[C]//. In: _2019 IEEE_ 

_International conference on big data (Big Data)_ , Los Angeles, CA, USA, pp. 3285–92. IEEE, 2019. 

23. Zhang M, Chen Y. Link prediction based on graph neural networks. _Adv Neural Inf Proces Syst_ 2018; **31** :5171–81. 

24. Kalman D. A singularly valuable decomposition: the SVD of a matrix. _Coll Math J_ 1996; **27** :2–23. https://doi.org/10.1080/ 07468342.1996.11973744. 

25. Dettweiler M, Reiter S. An algorithm of Katz and its application to the inverse Galois problem. _J Symb Comput_ 2000; **30** :761–98. https://doi.org/10.1006/jsco.2000.0382. 

26. Yu X, Jiang L, Jin S. _et al._ preMLI: a pre-trained method to uncover microRNA–lncRNA potential interactions. _Brief Bioinform_ 2022; **23** :bbab470. https://doi.org/10.1093/bib/bbab470. 

27. Kang Q, Meng J, Shi W. _et al._ Ensemble deep learning based on multi-level information enhancement and greedy fuzzy decision for plant miRNA–lncRNA interaction prediction. Interdisciplinary sciences: computational. _Life Sci_ 2021; **13** :603–14. https:// doi.org/10.1007/s12539-021-00434-7. 

28. Wang W, Zhang L, Sun J. _et al._ Predicting the potential human lncRNA–miRNA interactions based on graph convolution network with conditional random field. _Brief Bioinform_ 2022; **23** :bbac463. https://doi.org/10.1093/bib/bbac463. 

29. Kirk JM, Kim SO, Inoue K. _et al._ Functional classification of long non-coding RNAs by k-mer content. _Nat Genet_ 2018; **50** :1474–82. https://doi.org/10.1038/s41588-018-0207-8. 

30. Vaswani A, Shazeer N, Parmar N. _et al._ Attention is all you need. _Adv Neural Inf Proces Syst_ 2017; **30** :6000–10. 

31. Dong P, Xiong Y, Yue J. _et al._ Long non-coding RNA NEAT1: a novel target for diagnosis and therapy in human tumors. _Front Genet_ 2018; **9** :471. https://doi.org/10.3389/fgene.2018.00471. 

32. Standaert L, Adriaens C, Radaelli E. _et al._ The long noncoding RNA Neat1 is required for mammary gland development and lactation. _RNA_ 2014; **20** :1844–9. https://doi.org/10.1261/rna. 047332.114. 

33. Chen X, Kong J, Ma Z. _et al._ Up regulation of the long non-coding RNA NEAT1 promotes esophageal squamous cell carcinoma cell progression and correlates with poor prognosis. _Am J Cancer Res_ 2015; **75** :2808. https://doi.org/10.1158/1538-7445.AM2015-2808. 

34. Wang Q, Wang W, Zhang F. _et al._ NEAT1/miR-181c regulates osteopontin (OPN)-mediated synoviocyte proliferation in osteoarthritis. _J Cell Biochem_ 2017; **118** :3775–84. https://doi. org/10.1002/jcb.26025. 

35. Zhang F, Wu L, Qian J. _et al._ Identification of the long noncoding RNA NEAT1 as a novel inflammatory regulator acting through MAPK pathway in human lupus. _J Autoimmun_ 2016; **75** :96–104. https://doi.org/10.1016/j.jaut.2016.07.012. 

36. Wang Z, Zhao Y, Xu N. _et al._ NEAT1 regulates neuroglial cell mediating A _β_ clearance via the epigenetic regulation of endocytosis-related genes expression. _Cell Mol Life Sci_ 2019; **76** : 3005–18. https://doi.org/10.1007/s00018-019-03074-9. 

37. Lin W, Liu H, Tang Y. _et al._ The development and controversy of competitive endogenous RNA hypothesis in non-coding genes. _Mol Cell Biochem_ 2021; **476** :109–23. https://doi.org/10.1007/ s11010-020-03889-2. 

38. Miao YR, Liu W, Zhang Q. _et al._ lncRNASNP2: an updated database of functional SNPs and mutations in human and mouse lncRNAs. _Nucleic Acids Res_ 2018; **46** :D276–80. https://doi. org/10.1093/nar/gkx1004. 

39. O’Connor D. A historical note on shuffle algorithms. _Retrieved Maret_ 2014; **4** :2018. 

40. Frankish A, Diekhans M, Ferreira AM. _et al._ GENCODE reference annotation for the human and mouse genomes. _Nucleic Acids Res_ 2019; **47** :D766–73. https://doi.org/10.1093/nar/gky955. 

Predicting lncRNA–miRNA interactions with the TEC-LncMir | 15 

41. Kozomara A, Birgaoanu M, Griffiths-Jones S. miRBase: from microRNA sequences to function. _Nucleic Acids Res_ 2019; **47** :D155–62. https://doi.org/10.1093/nar/gky1141. 

42. Liu C, Bai B, Skogerbø G. _et al._ NONCODE: an integrated knowledge database of non-coding RNAs. _Nucleic Acids Res_ 2005; **33** :D112–5. https://doi.org/10.1093/nar/gki041. 

43. Clough E, Barrett T. The gene expression omnibus database. _Statistical Genomics: Methods and Protocols_ 2016; **1418** :93–110. https:// doi.org/10.1007/978-1-4939-3578-9_5. 

44. Harbig J, Sprinkle R, Enkemann SA. A sequence-based identification of the genes detected by probesets on the Affymetrix U133 plus 2.0 array. _Nucleic Acids Res_ 2005; **33** :e31–1. https://doi. org/10.1093/nar/gni027. 

45. Vo N, Hays J. Generalization in metric learning: Should the embedding layer be embedding layer?[C]//. In: _2019 IEEE winter conference on applications of computer vision (WACV)_ , Waikoloa, HI, USA, pp. 589–98. IEEE, 2019. 

46. Baldi P, Sadowski PJ. Understanding dropout. _Adv Neural Inf Proces Syst_ 2013; **26** :2814–22. 

47. Albrecht MR, Driessen B, Kavun EB. _et al._ Block ciphers–focus on the linear layer (feat. PRIDE)[C]//. In: _Advances in Cryptology– CRYPTO 2014: 34th Annual Cryptology Conference, Santa Barbara, CA, USA, August 17–21, 2014, Proceedings, Part I 34_ , Vol. **8616** , pp. 57–76. Springer Berlin Heidelberg. 

48. Nair V, Hinton GE. Rectified linear units improve restricted boltzmann machines[C]//. In: _Proceedings of the 27th international conference on machine learning (ICML-10)_ , Omnipress, Madison, WI, USA, pp. 807–14, 2010. 

49. Bjorck N, Gomes CP, Selman B. _et al._ Understanding batch normalization. _Adv Neural Inf Process Syst_ 2018; **31** :7705–16. 

50. Tommiska MT. Efficient digital implementation of the sigmoid function for reprogrammable logic. _IEE Proceedings-Computers and Digital Techniques_ 2003; **150** :403–11. https://doi.org/10.1049/ ip-cdt:20030965. 

51. Ruby U, Yendapalli V. Binary cross entropy with deep learning technique for image classification. _Int J Adv Trends Comput Sci Eng_ 2020; **9** :5393–7. https://doi.org/10.30534/ijatcse/2020/175942 020. 

52. Bock S, Weiß M. A proof of local convergence for the Adam optimizer[C]//. In: _2019 international joint conference on neural networks (IJCNN)_ , Budapest, Hungary, pp. 1–8. IEEE, 2019. 

53. Hossin M, Sulaiman MN. A review on evaluation metrics for data classification evaluations. _Int J Data Min Knowl Manag Process_ 2015; **5** :01–11. https://doi.org/10.5121/ijdkp.2015.5201. 

54. De Winter JCF. Using the Student’s t-test with extremely small sample sizes. _Pract Assess Res Eval_ 2019; **18** :10. 

55. Gene OC. Gene ontology consortium: going forward. _Nucleic Acids Res_ 2015; **43** :D1049–56. https://doi.org/10.1093/nar/gku1179. 

56. Kanehisa M, Goto S, Kawashima S. _et al._ The KEGG resource for deciphering the genome. _Nucleic Acids Res_ 2004; **32** :277D–80. https://doi.org/10.1093/nar/gkh063. 

57. Sedgwick P. Pearson’s correlation coefficient. _BMJ_ 2012; **345** : e4483. https://doi.org/10.1136/bmj.e4483. 

58. McGeary SE, Lin KS, Shi CY. _et al._ The biochemical basis of microRNA targeting efficacy. _Science_ 2019; **366** :eaav1741. https:// doi.org/10.1126/science.aav1741. 

59. Enright A, John B, Gaul U. _et al._ MicroRNA targets in drosophila. _Genome Biol_ 2003; **5** :1–27. https://doi.org/10.1186/ gb-2003-5-1-r1. 

60. Hoffman PW, Chernak JM. DNA binding and regulatory effects of transcription factors SP1 and USF at the rat amyloid precursor protein gene promoter. _Nucleic Acids Res_ 1995; **23** :2229–35. https:// doi.org/10.1093/nar/23.12.2229. 

61. Husseman JW, Hallows JL, Bregman DB. _et al._ Hyperphosphorylation of RNA polymerase II and reduced neuronal RNA levels precede neurofibrillary tangles in Alzheimer disease. _J Neuropathol Exp Neurol_ 2001; **60** :1219–32. https://doi.org/10.1093/ jnen/60.12.1219. 

62. Gong Y, Lippa CF. Disruption of the postsynaptic density in Alzheimer’s disease and other neurodegenerative dementias. _Am J Alzheimers Dis Other Dement_ 2010; **25** :547–55. https://doi. org/10.1177/1533317510382893. 

63. Kang K, Bai J, Zhong S. _et al._ Down-regulation of insulin like growth factor 1 involved in Alzheimer’s disease via MAPK, Ras, and FoxO Signaling pathways. _Oxidative Med Cell Longev_ 2022; **2022** :1–15. https://doi.org/10.1155/2022/8169981. 

64. Tang D, Chen M, Huang X. _et al._ SRplot: a free online platform for data visualization and graphing. _PLoS One_ 2023; **18** :e0294236. https://doi.org/10.1371/journal.pone.0294236. 

65. Zhang C, Gu Z, Shen L. _et al._ A dual targeting drug delivery system for penetrating blood-brain barrier and selectively delivering siRNA to neurons for Alzheimer’s disease treatment. _Curr Pharm Biotechnol_ 2017; **18** :1124–31. https://doi. org/10.2174/1389201019666180226152542. 

66. Zhang H, Li S, Dai N. _et al._ LinearCoFold and LinearCoPartition: linear-time algorithms for secondary structure prediction of interacting RNA molecules. _Nucleic Acids Res_ 2023; **51** :e94–4. https://doi.org/10.1093/nar/gkad664. 

© The Author(s) 2025. Published by Oxford University Press. This is an Open Access article distributed under the terms of the Creative Commons Attribution Non-Commercial License (https://creativecommons.org/licenses/by-nc/4.0/), which permits non-commercial re-use, distribution, and reproduction in any medium, provided the original work is properly cited. For commercial re-use, please contact journals.permissions@oup.com _Briefings in Bioinformatics_ , 2025, **26(1)** , bbaf046 https://doi.org/10.1093/bib/bbaf046 Problem Solving Protocol 

