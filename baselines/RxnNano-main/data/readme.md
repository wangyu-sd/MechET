# ReactionLLM

## Dataset
In the paper we use three open-source datasets (and train/valid/test splits).

* [**USPTO-50k** dataset](https://github.com/Hanjun-Dai/GLN):  
  * data/USPTO-50k/ (without reaction classes information) 
  *  data/USPTO-50k-typed/ (with reaction classes information)
* [**USPTO-full** dataset](https://github.com/Hanjun-Dai/GLN): 
  * data/USPTO-full/
* [**USPTO-480k/MIT** dataset](https://github.com/pschwllr/MolecularTransformer): 
  * data/USPTO-480k/

All of them are subsets from data extracted and originally published by Daniel Lowe

We do two main preprocessing methods:
* **Separated** reactants and reagents, since most of the reaction smile from the datasets are separated by a `>>` token, e.g. `O=C1CCC(=O)N1[Br:1].[CH3:2][c:3]1[cH:4][n:5][c:6]([Cl:7])[cH:8][c:9]1[Cl:10]>>[Br:1][CH2:2][c:3]1[cH:4][n:5][c:6]([Cl:7])[cH:8][c:9]1[Cl:10]`. We extracted the reactants, product and others important information and saved them in the format of jsonl file.
* **AAM-0** that replace every atom-mapping indices with a uniform placeholder “0”, e.g. `[Br:0][CH2:0][c:0]1[cH:0][n:0][c:0]([Cl:0])[cH:0][c:0]1[Cl:0]` 