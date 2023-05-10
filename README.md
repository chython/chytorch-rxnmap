Chytorch RxnMap
---------------

Semisupervised Model trained on USPTO and Pistachio datasets.

[Publication](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00344) ([Preprint](https://doi.org/10.26434/chemrxiv-2022-bn5nt)) with details.

Installation
------------

Use `pip install chytorch-rxnmap` to install release version.

Or `pip install .` in source code directory to install DEV version.

Perform Atom-to-atom mapping
----------------------------

AAM integrated into `chython` package and available as reaction object method. See `chython` documentation [here](https://chython.readthedocs.io).

    from chython import smiles

    r = smiles('OC(=O)C(=C)C=C.C=CC#N>>OC(=O)C1=CCCC(C1)C#N')
    r.reset_mapping()
    print(format(r, 'm'))
    >> [C:2]([C:4](=[CH2:5])[CH:6]=[CH2:7])(=[O:3])[OH:1].[CH2:8]=[CH:9][C:10]#[N:11]>>[O:3]=[C:2]([OH:1])[C:4]=1[CH2:5][CH:9]([C:10]#[N:11])[CH2:8][CH2:7][CH:6]=1
