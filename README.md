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


Pretrained model
----------------

**To load pretrained model use:**

    from chytorch.zoo.rxnmap import Model  
    model = Model.pretrained()

**To prepare data-loader use:**

    from chython import SMILESRead

    data = []
    for r in SMILESRead('data.smi'):
        r.canonicalize()  # fix aromaticity and functional groups
        data.append(r.pack())  # store in compressed format

    dl = model.prepare_dataloader(data, batch_size=20)

**To get embeddings use:**

    for b in dl:
        e = model(b)

Note: embeddings contain: `cls embedding, [unusable molecular embedding, list of atoms embeddings] * n`.
Where n is the number of molecules in reaction equation.

To extract aggregated embedding, use cls embedding `x = e[:, 0]`.

To extract atoms-only embeddings, use masking:
* `x = e[b[3] > 1]` - for all atoms
* `x = e[b[3] == 2]` - for reactants only
* `x = e[b[3] == 3]` - for products only

**To get all-to-all tokens attention matrix:**

    for b in dl:
        a = model(b, mapping_task=True)


Training new model
------------------

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins import DDPPlugin

    callback = ModelCheckpoint(monitor='trn_loss_tot', save_weights_only=True, save_top_k=3, save_last=True,
                               every_n_train_steps=10000)
    trainer = Trainer(gpus=-1, precision=16, max_steps=1000000, callbacks=[callback],
                      strategy=DDPPlugin(find_unused_parameters=False))

    model = Model(lr_warmup=1e4, lr_period=5e5, lr_max=1e-4, lr_decrease_coef=.01, masking_rate=.15, **kwargs)
    # lr_warmup=1e4, lr_period=5e5, lr_max=1e-4, lr_decrease_coef=.01 - see chytorch.optim.lr_scheduler.WarmUpCosine. 
    # kwargs - see chytorch.nn.ReactionEncoder.
    # masking_rate - probability of token masking.
    trainer.fit(model, dl)
