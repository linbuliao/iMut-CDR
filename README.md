# iMut-CDR

**iMut-CDR** (Iterative Mutator of Antibody CDRs) is a computational framework that frames antibody complementarity-determining region (CDR) optimization as a masked-residue recovery problem.  
By combining cross-entropy learning with contrastive alignment and token-level self-distillation, iMut-CDR captures both local residue context and global sequence-level constraints. At inference time, it applies mutations iteratively—one site at a time—so that proposed substitutions naturally respect co-evolutionary dependencies among positions.

## Pretrained Model

A pretrained model checkpoint (**best.pt**) is available here:

[Download best.pt](https://drive.google.com/file/d/1mLfoSNwKDw0c9Fmc1ajxK7nrHLgFSKp-/view?usp=sharing)

## Repository Scope

This repository provides the **code for applying iMut-CDR to perform in-silico mutagenesis of antibody CDRs**.  
It is focused on running mutations with the pretrained model and does not include training data or training scripts.
