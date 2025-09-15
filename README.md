# iMut-CDR

**iMut-CDR** (Iterative Mutator of Antibody CDRs) is a computational framework for **in-silico optimization of antibody complementarity-determining regions (CDRs)**.  
The method formulates CDR refinement as a masked-residue recovery problem. By combining cross-entropy learning with contrastive alignment and token-level self-distillation, iMut-CDR captures both local residue context and global sequence-level constraints. During inference, mutations are applied iteratively—one site at a time—so that each substitution naturally respects co-evolutionary dependencies among positions.

## Pretrained Model

A pretrained model checkpoint (**best.pt**) is available for download:

[Download best.pt](https://drive.google.com/file/d/1mLfoSNwKDw0c9Fmc1ajxK7nrHLgFSKp-/view?usp=sharing)

## Repository Scope

This repository provides the **code for applying iMut-CDR to perform in-silico mutagenesis of antibody CDRs** using the pretrained model.  
It is focused on mutation and inference workflows.

## Usage

The main entry point is **`mutate_v5.py`**, which performs in-silico mutagenesis for antibody CDRs.  
Examples of how to run the script are included at the end of `mutate_v5.py`.  

To get started:
```bash
python mutate_v5.py
