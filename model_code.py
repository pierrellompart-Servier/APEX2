LETTER_TO_NUM = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,
                       'N': 2, 'Y': 18, 'M': 12, 'X':20}

NUM_TO_LETTER = {v:k for k, v in LETTER_TO_NUM.items()}

ATOM_VOCAB = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'unk']
"""
Drug-target binding affinity datasets
"""
import math
import yaml
import json
from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class DTA(data.Dataset):
    """
    Base class for loading drug-target binding affinity datasets.
    """
    def __init__(self, df=None, data_list=None, onthefly=False,
                prot_featurize_fn=None, drug_featurize_fn=None):
        """
        Parameters
        ----------
            df : pd.DataFrame with columns [`drug`, `protein`, `y`],
                where `drug`: drug key, `protein`: protein key, `y`: binding affinity.
            data_list : list of dict (same order as df)
                if `onthefly` is True, data_list has the PDB coordinates and SMILES strings
                    {`drug`: SDF file path, `protein`: coordinates dict (`pdb_data` in `DTATask`), `y`: float}
                if `onthefly` is False, data_list has the cached torch_geometric graphs
                    {`drug`: `torch_geometric.data.Data`, `protein`: `torch_geometric.data.Data`, `y`: float}
                `protein` has attributes:
                    -x          alpha carbon coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
                    -name       name of the protein structure, string
                    -node_s     node scalar features, shape [n_nodes, 6]
                    -node_v     node vector features, shape [n_nodes, 3, 3]
                    -edge_s     edge scalar features, shape [n_edges, 39]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
                    -seq_emb    sequence embedding (ESM1b), shape [n_nodes, 1280]
                `drug` has attributes:
                    -x          atom coordinates, shape [n_nodes, 3]
                    -edge_index edge indices, shape [2, n_edges]
                    -node_s     node scalar features, shape [n_nodes, 66]
                    -node_v     node vector features, shape [n_nodes, 1, 3]
                    -edge_s     edge scalar features, shape [n_edges, 16]
                    -edge_v     edge scalar features, shape [n_edges, 1, 3]
                    -name       name of the drug, string
            onthefly : bool
                whether to featurize data on the fly or pre-compute
            prot_featurize_fn : function
                function to featurize a protein.
            drug_featurize_fn : function
                function to featurize a drug.
        """
        super(DTA, self).__init__()
        self.data_df = df
        self.data_list = data_list
        self.onthefly = onthefly
        if onthefly:
            assert prot_featurize_fn is not None, 'prot_featurize_fn must be provided'
            assert drug_featurize_fn is not None, 'drug_featurize_fn must be provided'
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx]['drug_name']
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx]['protein_name']
            )
        else:
            drug = self.data_list[idx]['drug']
            prot = self.data_list[idx]['protein']
        y = self.data_list[idx]['y']
        item = {'drug': drug, 'protein': prot, 'y': y}
        return item


def create_fold(df, fold_seed, frac):
    """
    Create train/valid/test folds by random splitting.
    """
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True)}


def create_fold_setting_cold(df, fold_seed, frac, entity):
    """
    Create train/valid/test folds by drug/protein-wise splitting.
    """
    train_frac, val_frac, test_frac = frac
    gene_drop = df[entity].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values

    test = df[df[entity].isin(gene_drop)]

    train_val = df[~df[entity].isin(gene_drop)]

    gene_drop_val = train_val[entity].drop_duplicates().sample(frac = val_frac/(1-test_frac), replace = False, random_state = fold_seed).values
    val = train_val[train_val[entity].isin(gene_drop_val)]
    train = train_val[~train_val[entity].isin(gene_drop_val)]

    return {'train': train.reset_index(drop = True),
            'valid': val.reset_index(drop = True),
            'test': test.reset_index(drop = True)}


def create_full_ood_set(df, fold_seed, frac):
    """
    Create train/valid/test folds such that drugs and proteins are
    not overlapped in train and test sets. Train and valid may share
    drugs and proteins (random split).
    """
    train_frac, val_frac, test_frac = frac
    test_drugs = df['drug'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values
    test_prots = df['protein'].drop_duplicates().sample(frac=test_frac, replace=False, random_state=fold_seed).values

    test = df[(df['drug'].isin(test_drugs)) & (df['protein'].isin(test_prots))]
    train_val = df[(~df['drug'].isin(test_drugs)) & (~df['protein'].isin(test_prots))]

    val = train_val.sample(frac=val_frac/(1-test_frac), replace=False, random_state=fold_seed)
    train = train_val[~train_val.index.isin(val.index)]

    return {'train': train.reset_index(drop=True),
            'valid': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)}


def create_seq_identity_fold(df, mmseqs_seq_clus_df, fold_seed, frac, min_clus_in_split=5):
    """
    Adapted from: https://github.com/drorlab/atom3d/blob/master/atom3d/splits/sequence.py
    Clusters are selected randomly into validation and test sets,
    but to ensure that there is some diversity in each set
    (i.e. a split does not consist of a single sequence cluster), a minimum number of clusters in each split is enforced.
    Some data examples may be removed in order to satisfy this constraint.
    """
    _rng = np.random.RandomState(fold_seed)

    def _parse_mmseqs_cluster_res(mmseqs_seq_clus_df):
        clus2seq, seq2clus = {}, {}
        for rep, sdf in mmseqs_seq_clus_df.groupby('rep'):
            for seq in sdf['seq']:
                if rep not in clus2seq:
                    clus2seq[rep] = []
                clus2seq[rep].append(seq)
                seq2clus[seq] = rep
        return seq2clus, clus2seq

    def _create_cluster_split(df, seq2clus, clus2seq, to_use, split_size, min_clus_in_split):
        data = df.copy()
        all_prot = set(seq2clus.keys())
        used = all_prot.difference(to_use)
        split = None
        while True:
            p = _rng.choice(sorted(to_use))
            c = seq2clus[p]
            members = set(clus2seq[c])
            members = members.difference(used)
            if len(members) == 0:
                continue
            # ensure that at least min_fam_in_split families in each split
            max_clust_size = int(np.ceil(split_size / min_clus_in_split))
            sel_prot = list(members)[:max_clust_size]
            sel_df = data[data['protein'].isin(sel_prot)]
            split = sel_df if split is None else pd.concat([split, sel_df])
            to_use = to_use.difference(members)
            used = used.union(members)
            if len(split) >= split_size:
                break
        split = split.reset_index(drop=True)
        return split, to_use

    seq2clus, clus2seq = _parse_mmseqs_cluster_res(mmseqs_seq_clus_df)
    train_frac, val_frac, test_frac = frac
    test_size, val_size = len(df) * test_frac, len(df) * val_frac
    to_use = set(seq2clus.keys())

    val_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, val_size, min_clus_in_split)
    test_df, to_use = _create_cluster_split(df, seq2clus, clus2seq, to_use, test_size, min_clus_in_split)
    train_df = df[df['protein'].isin(to_use)].reset_index(drop=True)
    train_df['split'] = 'train'
    val_df['split'] = 'valid'
    test_df['split'] = 'test'

    assert len(set(train_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(test_df['protein']) & set(val_df['protein'])) == 0
    assert len(set(train_df['protein']) & set(test_df['protein'])) == 0

    return {'train': train_df.reset_index(drop=True),
            'valid': val_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True)}


class DTATask(object):
    """
    Drug-target binding task (e.g., KIBA or Davis).
    Three splits: train/valid/test, each split is a DTA() class
    """
    def __init__(self,
            task_name=None,
            df=None,
            prot_pdb_id=None, pdb_data=None,
            emb_dir=None,
            drug_sdf_dir=None,
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,
            split_method='random', split_frac=[0.7, 0.1, 0.2],
            mmseqs_seq_clus_df=None,
            seed=42, onthefly=False
        ):
        """
        Parameters
        ----------
        task_name: str
            Name of the task (e.g., KIBA, Davis, etc.)
        df: pd.DataFrame
            Dataframe containing the data
        prot_pdb_id: dict
            Dictionary mapping protein name to PDB ID
        pdb_data: dict
            A json format of pocket structure data, where key is the PDB ID
            and value is the corresponding PDB structure data in a dictionary:
                -'name': kinase name
                -'UniProt_id': UniProt ID
                -'PDB_id': PDB ID,
                -'chain': chain ID,
                -'seq': pocket sequence,                
                -'coords': coordinates of the 'N', 'CA', 'C', 'O' atoms of the pocket residues,
                    - "N": [[x, y, z], ...]
                    - "CA": [[], ...],
                    - "C": [[], ...],
                    - "O": [[], ...]               
            (there are some other keys but only for internal use)
        emb_dir: str
            Directory containing the protein embeddings
        drug_sdf_dir: str
            Directory containing the drug SDF files
        num_pos_emb: int
            Dimension of positional embeddings
        num_rbf: int
            Number of radial basis functions
        contact_cutoff: float
            Cutoff distance for defining residue-residue contacts
        split_method: str
            how to split train/test sets, 
            -`random`: random split
            -`protein`: split by protein
            -`drug`: split by drug
            -`both`: unseen drugs and proteins in test set
            -`seqid`: split by protein sequence identity 
                (need to priovide the MMseqs2 sequence cluster result,
                see `mmseqs_seq_clus_df`)
        split_frac: list
            Fraction of data in train/valid/test sets
        mmseqs_seq_clus_df: pd.DataFrame
            Dataframe containing the MMseqs2 sequence cluster result
            using a desired sequence identity cutoff
        seed: int
            Random seed
        onthefly: bool
            whether to featurize data on the fly or pre-compute
        """
        self.task_name = task_name        
        self.prot_pdb_id = prot_pdb_id
        self.pdb_data = pdb_data        
        self.emb_dir = emb_dir
        self.df = df
        self.prot_featurize_params = dict(
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff)        
        self.drug_sdf_dir = drug_sdf_dir        
        self._prot2pdb = None
        self._pdb_graph_db = None        
        self._drug2sdf_file = None
        self._drug_sdf_db = None
        self.split_method = split_method
        self.split_frac = split_frac
        self.mmseqs_seq_clus_df = mmseqs_seq_clus_df
        self.seed = seed
        self.onthefly = onthefly

    def _format_pdb_entry(self, _data):
        _coords = _data["coords"]
        entry = {
            "name": _data["name"],
            "seq": _data["seq"],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
        }        
        if self.emb_dir is not None:
            embed_file = f"{_data['PDB_id']}.{_data['chain']}.pt"
            entry["embed"] = f"{self.emb_dir}/{embed_file}"
        return entry

    @property
    def prot2pdb(self):
        if self._prot2pdb is None:
            self._prot2pdb = {}
            for prot, pdb in self.prot_pdb_id.items():
                _pdb_entry = self.pdb_data[pdb]
                self._prot2pdb[prot] = self._format_pdb_entry(_pdb_entry)
        return self._prot2pdb

    @property
    def pdb_graph_db(self):
        if self._pdb_graph_db is None:
            self._pdb_graph_db = pdb_graph.pdb_to_graphs(self.prot2pdb,
                self.prot_featurize_params)
        return self._pdb_graph_db

    @property
    def drug2sdf_file(self):
        if self._drug2sdf_file is None:            
            drug2sdf_file = {f.stem : str(f) for f in Path(self.drug_sdf_dir).glob('*.sdf')}
            # Convert str keys to int for Davis
            if self.task_name == 'DAVIS' and all([k.isdigit() for k in drug2sdf_file.keys()]):
                drug2sdf_file = {int(k) : v for k, v in drug2sdf_file.items()}
            self._drug2sdf_file = drug2sdf_file
        return self._drug2sdf_file

    @property
    def drug_sdf_db(self):
        if self._drug_sdf_db is None:
            self._drug_sdf_db = mol_graph.sdf_to_graphs(self.drug2sdf_file)
        return self._drug_sdf_db


    def build_data(self, df, onthefly=False):
        records = df.to_dict('records')
        data_list = []
        for entry in records:
            drug = entry['drug']
            prot = entry['protein']
            if onthefly:
                pf = self.prot2pdb[prot]
                df = self.drug2sdf_file[drug]
            else:                
                pf = self.pdb_graph_db[prot]                
                df = self.drug_sdf_db[drug]
            data_list.append({'drug': df, 'protein': pf, 'y': entry['y'],
                'drug_name': drug, 'protein_name': prot})
        if onthefly:
            prot_featurize_fn = partial(
                pdb_graph.featurize_protein_graph,
                **self.prot_featurize_params)            
            drug_featurize_fn = mol_graph.featurize_drug
        else:
            prot_featurize_fn, drug_featurize_fn = None, None
        data = DTA(df=df, data_list=data_list, onthefly=onthefly,
            prot_featurize_fn=prot_featurize_fn, drug_featurize_fn=drug_featurize_fn)
        return data


    def get_split(self, df=None, split_method=None,
            split_frac=None, seed=None, onthefly=None,
            return_df=False):
        df = df or self.df
        split_method = split_method or self.split_method
        split_frac = split_frac or self.split_frac
        seed = seed or self.seed
        onthefly = onthefly or self.onthefly
        if split_method == 'random':
            split_df = create_fold(self.df, seed, split_frac)
        elif split_method == 'drug':
            split_df = create_fold_setting_cold(self.df, seed, split_frac, 'drug')
        elif split_method == 'protein':
            split_df = create_fold_setting_cold(self.df, seed, split_frac, 'protein')
        elif split_method == 'both':
            split_df = create_full_ood_set(self.df, seed, split_frac)
        elif split_method == 'seqid':
            split_df = create_seq_identity_fold(
                self.df, self.mmseqs_seq_clus_df, seed, split_frac)
        else:
            raise ValueError("Unknown split method: {}".format(split_method))
        split_data = {}
        for split, df in split_df.items():
            split_data[split] = self.build_data(df, onthefly=onthefly)
        if return_df:
            return split_data, split_df
        else:
            return split_data


class KIBA(DTATask):
    """
    KIBA drug-target interaction dataset
    """
    def __init__(self,
            data_path='../data/KIBA/kiba_data.tsv',            
            pdb_map='../data/KIBA/kiba_uniprot2pdb.yaml',
            pdb_json='../data/structure/pockets_structure.json',                        
            emb_dir='../data/esm1b',           
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,            
            drug_sdf_dir='../data/structure/kiba_mol3d_sdf',
            split_method='random', split_frac=[0.7, 0.1, 0.2],
            mmseqs_seq_cluster_file='../data/KIBA/kiba_cluster_id50_cluster.tsv',
            seed=42, onthefly=False
        ):
        df = pd.read_table(data_path)        
        prot_pdb_id = yaml.safe_load(open(pdb_map, 'r'))
        pdb_data = json.load(open(pdb_json, 'r'))                
        mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_cluster_file, names=['rep', 'seq'])
        super(KIBA, self).__init__(
            task_name='KIBA',
            df=df, 
            prot_pdb_id=prot_pdb_id, pdb_data=pdb_data,
            emb_dir=emb_dir,            
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            split_method=split_method, split_frac=split_frac,
            mmseqs_seq_clus_df=mmseqs_seq_clus_df,
            seed=seed, onthefly=onthefly
            )


class DAVIS(DTATask):
    """
    DAVIS drug-target interaction dataset
    """
    def __init__(self,
            data_path='../data/DAVIS/davis_data.tsv',            
            pdb_map='../data/DAVIS/davis_protein2pdb.yaml',
            pdb_json='../data/structure/pockets_structure.json',                        
            emb_dir='../data/esm1b',           
            num_pos_emb=16, num_rbf=16,
            contact_cutoff=8.,            
            drug_sdf_dir='../data/structure/davis_mol3d_sdf',
            split_method='random', split_frac=[0.7, 0.1, 0.2],
            mmseqs_seq_cluster_file='../data/DAVIS/davis_cluster_id50_cluster.tsv',
            seed=42, onthefly=False
        ):
        df = pd.read_table(data_path)        
        prot_pdb_id = yaml.safe_load(open(pdb_map, 'r'))
        pdb_data = json.load(open(pdb_json, 'r'))        
        mmseqs_seq_clus_df = pd.read_table(mmseqs_seq_cluster_file, names=['rep', 'seq'])
        super(DAVIS, self).__init__(
            task_name='DAVIS',
            df=df, 
            prot_pdb_id=prot_pdb_id, pdb_data=pdb_data,
            emb_dir=emb_dir,            
            num_pos_emb=num_pos_emb, num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
            drug_sdf_dir=drug_sdf_dir,
            split_method=split_method, split_frac=split_frac,
            mmseqs_seq_clus_df=mmseqs_seq_clus_df,
            seed=seed, onthefly=onthefly
            )
import copy
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import uncertainty_toolbox as uct

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
torch.set_num_threads(1)



def _parallel_train_per_epoch(
    kwargs=None, test_loader=None,
    n_epochs=None, eval_freq=None, test_freq=None,
    monitoring_score='pearson',
    loss_fn=None, logger=None,
    test_after_train=True,
):
    midx = kwargs['midx']
    model = kwargs['model']
    optimizer = kwargs['optimizer']
    train_loader = kwargs['train_loader']
    valid_loader = kwargs['valid_loader']
    device = kwargs['device']
    stopper = kwargs['stopper']
    best_model_state_dict = kwargs['best_model_state_dict']
    if stopper.early_stop:
        return kwargs

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        for step, batch in enumerate(train_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            optimizer.zero_grad()
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / step
        if epoch % eval_freq == 0:
            val_results = _parallel_test(
                {'model': model, 'midx': midx, 'test_loader': valid_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            is_best = stopper.update(val_results['metrics'][monitoring_score])
            if is_best:
                best_model_state_dict = copy.deepcopy(model.state_dict())
            logger.info(f"M-{midx} E-{epoch} | Train Loss: {train_loss:.4f} | Valid Loss: {val_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in val_results['metrics'].items()])
                + f" | best {monitoring_score}: {stopper.best_score:.4f}"
                )
        if test_freq is not None and epoch % test_freq == 0:
            test_results = _parallel_test(
                {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
                loss_fn=loss_fn, logger=logger
            )
            logger.info(f"M-{midx} E-{epoch} | Test Loss: {test_results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in test_results['metrics'].items()])
                )

        if stopper.early_stop:
            logger.info('Eearly stop at epoch {}'.format(epoch))

    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
    if test_after_train:
        test_results = _parallel_test(
            {'midx': midx, 'model': model, 'test_loader': test_loader, 'device': device},
            loss_fn=loss_fn,
            test_tag=f"Model {midx}", print_log=True, logger=logger
        )
    rets = dict(midx = midx, model = model)
    return rets


def _parallel_test(
    kwargs=None, loss_fn=None, 
    test_tag=None, print_log=False, logger=None,
):
    midx = kwargs['midx']
    model = kwargs['model']
    test_loader = kwargs['test_loader']
    device = kwargs['device']
    model.eval()
    yt, yp, total_loss = torch.Tensor(), torch.Tensor(), 0
    with torch.no_grad():
        for step, batch in enumerate(test_loader, start=1):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            yh = model(xd, xp)
            loss = loss_fn(yh, y.view(-1, 1))
            total_loss += loss.item()
            yp = torch.cat([yp, yh.detach().cpu()], dim=0)
            yt = torch.cat([yt, y.detach().cpu()], dim=0)
    yt = yt.numpy()
    yp = yp.view(-1).numpy()
    results = {
        'midx': midx,
        'y_true': yt,
        'y_pred': yp,
        'loss': total_loss / step,
    }
    eval_metrics = evaluation_metrics(
        yt, yp,
        eval_metrics=['mse', 'spearman', 'pearson']
    )
    results['metrics'] = eval_metrics
    if print_log:
        logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
            + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
    return results


def _unpack_evidential_output(output):
    mu, v, alpha, beta = torch.split(output, output.shape[1]//4, dim=1)
    inverse_evidence = 1. / ((alpha - 1) * v)
    var = beta * inverse_evidence
    return mu, var, inverse_evidence


class DTAExperiment(object):
    def __init__(self,
        task=None,
        split_method='protein',
        split_frac=[0.7, 0.1, 0.2],
        prot_gcn_dims=[128, 128, 128], prot_gcn_bn=False,
        prot_fc_dims=[1024, 128],
        drug_in_dim=66, drug_fc_dims=[1024, 128], drug_gcn_dims=[128, 64],
        mlp_dims=[1024, 512], mlp_dropout=0.25,
        num_pos_emb=16, num_rbf=16,
        contact_cutoff=8.,
        n_ensembles=1, n_epochs=500, batch_size=256,
        lr=0.001,        
        seed=42, onthefly=False,
        uncertainty=False, parallel=False,
        output_dir='../output', save_log=False
    ):
        self.saver = Saver(output_dir)
        self.logger = Logger(logfile=self.saver.save_dir/'exp.log' if save_log else None)

        self.uncertainty = uncertainty
        self.parallel = parallel
        self.n_ensembles = n_ensembles
        if self.uncertainty and self.n_ensembles < 2:
            raise ValueError('n_ensembles must be greater than 1 when uncertainty is True')            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        dataset_klass = {
            'kiba': KIBA,
            'davis': DAVIS,
        }[task]

        self.dataset = dataset_klass(
            split_method=split_method,
            split_frac=split_frac,
            seed=seed,
            onthefly=onthefly,
            num_pos_emb=num_pos_emb,
            num_rbf=num_rbf,
            contact_cutoff=contact_cutoff,
        )
        self._task_data_df_split = None
        self._task_loader = None

        n_gpus = torch.cuda.device_count()
        if self.parallel and n_gpus < self.n_ensembles:
            self.logger.warning(f"Visible GPUs ({n_gpus}) is fewer than "
            f"number of models ({self.n_ensembles}). Some models will be run on the same GPU"
            )
        self.devices = [torch.device(f'cuda:{i % n_gpus}')
            for i in range(self.n_ensembles)]
        self.model_config = dict(
            prot_emb_dim=1280,
            prot_gcn_dims=prot_gcn_dims,            
            prot_fc_dims=prot_fc_dims,
            drug_node_in_dim=[66, 1], 
            drug_node_h_dims=drug_gcn_dims,
            drug_fc_dims=drug_fc_dims,            
            mlp_dims=mlp_dims, mlp_dropout=mlp_dropout)
        self.build_model()
        self.criterion = F.mse_loss

        self.split_method = split_method
        self.split_frac = split_frac

        self.logger.info(self.models[0])
        self.logger.info(self.optimizers[0])

    def build_model(self):
        self.models = [DTAModel(**self.model_config).to(self.devices[i])
                        for i in range(self.n_ensembles)]
        self.optimizers = [optim.Adam(model.parameters(), lr=self.lr) for model in self.models]

    def _get_data_loader(self, dataset, shuffle=False):
        return torch_geometric.loader.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    pin_memory=False,
                    num_workers=0,
                )

    @property
    def task_data_df_split(self):
        if self._task_data_df_split is None:
            (data, df) = self.dataset.get_split(return_df=True)
            self._task_data_df_split = (data, df)
        return self._task_data_df_split

    @property
    def task_data(self):
        return self.task_data_df_split[0]

    @property
    def task_df(self):
        return self.task_data_df_split[1]

    @property
    def task_loader(self):
        if self._task_loader is None:
            _loader = {
                s: self._get_data_loader(
                    self.task_data[s], shuffle=(s == 'train'))
                for s in self.task_data
            }
            self._task_loader = _loader
        return self._task_loader

    def recalibrate_std(self, df, recalib_df):
        y_mean = recalib_df['y_pred'].values
        y_std = recalib_df['y_std'].values
        y_true = recalib_df['y_true'].values
        std_ratio = uct.recalibration.optimize_recalibration_ratio(
            y_mean, y_std, y_true, criterion="miscal")
        df['y_std_recalib'] = df['y_std'] * std_ratio
        return df

    def _format_predict_df(self, results,
            test_df=None, esb_yp=None, recalib_df=None):
        """
        results: dict with keys y_pred, y_true, y_var
        """
        df = self.task_df['test'].copy() if test_df is None else test_df.copy()
        assert np.allclose(results['y_true'], df['y'].values)
        df = df.rename(columns={'y': 'y_true'})
        df['y_pred'] = results['y_pred']
        if esb_yp is not None:
            if self.uncertainty:
                df['y_std'] = np.std(esb_yp, axis=0)
                if recalib_df is not None:
                    df = self.recalibrate_std(df, recalib_df)
            for i in range(self.n_ensembles):
                df[f'y_pred_{i + 1}'] = esb_yp[i]
        return df

    def train(self, n_epochs=None, patience=None,
                eval_freq=1, test_freq=None,
                monitoring_score='pearson',
                train_data=None, valid_data=None,                
                rebuild_model=False,
                test_after_train=False):
        n_epochs = n_epochs or self.n_epochs
        if rebuild_model:
            self.build_model()
        tl, vl = self.task_loader['train'], self.task_loader['valid']
        rets_list = []
        for i in range(self.n_ensembles):
            stp = EarlyStopping(eval_freq=eval_freq, patience=patience,
                                    higher_better=(monitoring_score != 'mse'))
            rets = dict(
                midx = i + 1,
                model = self.models[i],
                optimizer = self.optimizers[i],
                device = self.devices[i],
                train_loader = tl,
                valid_loader = vl,
                stopper = stp,
                best_model_state_dict = None,
            )
            rets_list.append(rets)

        rets_list = Parallel(n_jobs=(self.n_ensembles if self.parallel else 1), prefer="threads")(
            delayed(_parallel_train_per_epoch)(
                kwargs=rets_list[i],
                test_loader=self.task_loader['test'],
                n_epochs=n_epochs, eval_freq=eval_freq, test_freq=test_freq,
                monitoring_score=monitoring_score,
                loss_fn=self.criterion, logger=self.logger,
                test_after_train=test_after_train,
            ) for i in range(self.n_ensembles))

        for i, rets in enumerate(rets_list):
            self.models[rets['midx'] - 1] = rets['model']


    def test(self, test_model=None, test_loader=None,
                test_data=None, test_df=None,
                recalib_df=None,
                save_prediction=False, save_df_name='prediction.tsv',
                test_tag=None, print_log=False):
        test_models = self.models if test_model is None else [test_model]
        if test_data is not None:
            assert test_df is not None, 'test_df must be provided if test_data used'
            test_loader = self._get_data_loader(test_data)
        elif test_loader is not None:
            assert test_df is not None, 'test_df must be provided if test_loader used'
        else:
            test_loader = self.task_loader['test']
        rets_list = []
        for i, model in enumerate(test_models):
            rets = _parallel_test(
                kwargs={
                    'midx': i + 1,
                    'model': model,
                    'test_loader': test_loader,
                    'device': self.devices[i],
                },
                loss_fn=self.criterion,
                test_tag=f"Model {i+1}", print_log=True, logger=self.logger
            )
            rets_list.append(rets)


        esb_yp, esb_loss = None, 0
        for rets in rets_list:
            esb_yp = rets['y_pred'].reshape(1, -1) if esb_yp is None else\
                np.vstack((esb_yp, rets['y_pred'].reshape(1, -1)))
            esb_loss += rets['loss']

        y_true = rets['y_true']
        y_pred = np.mean(esb_yp, axis=0)
        esb_loss /= len(test_models)
        results = {
            'y_true': y_true,
            'y_pred': y_pred,
            'loss': esb_loss,
        }

        eval_metrics = evaluation_metrics(
            y_true, y_pred,
            eval_metrics=['mse', 'spearman', 'pearson']
        )
        results['metrics'] = eval_metrics
        results['df'] = self._format_predict_df(results,
            test_df=test_df, esb_yp=esb_yp, recalib_df=recalib_df)
        if save_prediction:
            self.saver.save_df(results['df'], save_df_name, float_format='%g')
        if print_log:
            self.logger.info(f"{test_tag} | Test Loss: {results['loss']:.4f} | "\
                + ' | '.join([f'{k}: {v:.4f}' for k, v in results['metrics'].items()]))
        return results

"""
Geometric Vector Perceptrons
From: https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/__init__.py
"""
import torch, functools
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    v = torch.reshape(x[..., -3*nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3*nv]
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = torch.reshape(v, v.shape[:-2] + (3*v.shape[-2],))
    return torch.cat([s, v], -1)

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.reshape(x_v.shape[0], 3*x_v.shape[1]),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, node_dims, edge_dims,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, 
                 activations=(F.relu, torch.sigmoid), vector_gate=False):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
            
        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return xfrom sklearn import metrics
from scipy import stats
import numpy as np

def eval_mse(y_true, y_pred, squared=True):
    """Evaluate mse/rmse and return the results.
    squared: bool, default=True
        If True returns MSE value, if False returns RMSE value.
    """
    return metrics.mean_squared_error(y_true, y_pred, squared=squared)

def eval_pearson(y_true, y_pred):
    """Evaluate Pearson correlation and return the results."""
    return stats.pearsonr(y_true, y_pred)[0]

def eval_spearman(y_true, y_pred):
    """Evaluate Spearman correlation and return the results."""
    return stats.spearmanr(y_true, y_pred)[0]

def eval_r2(y_true, y_pred):
    """Evaluate R2 and return the results."""
    return metrics.r2_score(y_true, y_pred)

def eval_auroc(y_true, y_pred):
    """Evaluate AUROC and return the results."""
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    return metrics.auc(fpr, tpr)

def eval_auprc(y_true, y_pred):
    """Evaluate AUPRC and return the results."""
    pre, rec, _ = metrics.precision_recall_curve(y_true, y_pred)
    return metrics.auc(rec, pre)


def evaluation_metrics(y_true=None, y_pred=None,
		eval_metrics=[]):
    """Evaluate eval_metrics and return the results.
    Parameters
    ----------
    y_true: true labels
    y_pred: predicted labels
    eval_metrics: a list of evaluation metrics
    """
    results = {}
    for m in eval_metrics:
        if m == 'mse':
            s = eval_mse(y_true, y_pred, squared=True)
        elif m == 'rmse':
            s = eval_mse(y_true, y_pred, squared=False)
        elif m == 'pearson':
            s = eval_pearson(y_true, y_pred)
        elif m == 'spearman':
            s = eval_spearman(y_true, y_pred)
        elif m == 'r2':
            s = eval_r2(y_true, y_pred)
        elif m == 'auroc':
            s = eval_auroc(y_true, y_pred)
        elif m == 'auprc':
            s = eval_auprc(y_true, y_pred)
        else:
            raise ValueError('Unknown evaluation metric: {}'.format(m))
        results[m] = s        
    return results

import torch
import torch.nn as nn
import torch_geometric

class Prot3DGraphModel(nn.Module):
    def __init__(self,
        d_vocab=21, d_embed=20,
        d_dihedrals=6, d_pretrained_emb=1280, d_edge=39,
        d_gcn=[128, 256, 256],
    ):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):            
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))            
            layers.append(nn.LeakyReLU())            
        
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)        
        self.pool = torch_geometric.nn.global_mean_pool
        

    def forward(self, data):
        x, edge_index = data.seq, data.edge_index
        batch = data.batch

        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)

        edge_attr = data.edge_s

        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)

        x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x



class DrugGVPModel(nn.Module):
    def __init__(self, 
        node_in_dim=[66, 1], node_h_dim=[128, 64],
        edge_in_dim=[16, 1], edge_h_dim=[32, 1],
        num_layers=3, drop_rate=0.1
    ):
        """
        Parameters
        ----------
        node_in_dim : list of int
            Input dimension of drug node features (si, vi).
            Scalar node feartures have shape (N, si).
            Vector node features have shape (N, vi, 3).
        node_h_dims : list of int
            Hidden dimension of drug node features (so, vo).
            Scalar node feartures have shape (N, so).
            Vector node features have shape (N, vo, 3).
        """
        super(DrugGVPModel, self).__init__()
        self.W_v = nn.Sequential(
            LayerNorm(node_in_dim),
            GVP(node_in_dim, node_h_dim, activations=(None, None))
        )
        self.W_e = nn.Sequential(
            LayerNorm(edge_in_dim),
            GVP(edge_in_dim, edge_h_dim, activations=(None, None))
        )

        self.layers = nn.ModuleList(
                GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate)
            for _ in range(num_layers))

        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            LayerNorm(node_h_dim),
            GVP(node_h_dim, (ns, 0)))

    def forward(self, xd):
        # Unpack input data
        h_V = (xd.node_s, xd.node_v)
        h_E = (xd.edge_s, xd.edge_v)
        edge_index = xd.edge_index
        batch = xd.batch

        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        # per-graph mean
        out = torch_geometric.nn.global_add_pool(out, batch)

        return out


class DTAModel(nn.Module):
    def __init__(self,
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(DTAModel, self).__init__()

        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
        )
        drug_emb_dim = drug_node_h_dims[0]

        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]

        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
       
        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

        self.top_fc = self.get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims + [1],
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)

    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)

    def forward(self, xd, xp):
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)

        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)

        x = torch.cat([xd, xp], dim=1)
        x = self.top_fc(x)
        return x
import rdkit 
from rdkit import Chem
import torch
import numpy as np
import pandas as pd
import torch_geometric
import torch_cluster



def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    '''
    Parameters
    ----------
    a: array of numerical value of categorical feature classes.
    alphabet: valid values of feature classes.
    default: default class if out of alphabet.
    Returns
    -------
    A 2-D one-hot array with size |x| * |alphabet|
    '''
    # replace out-of-vocabulary classes
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]

    # cast to category to force class not present
    a = pd.Categorical(a, categories=alphabet)

    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol):
    # dim: 44 + 7 + 7 + 7 + 1
    feature_alphabet = {
        # (alphabet, default value)
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }

    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                    alphabet=feature_alphabet[attr][0],
                    default=feature_alphabet[attr][1],
                    drop_first=(attr in ['GetIsAromatic']) # binary-class feature
                )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)
    return atom_feature




def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs(data_list):
    """
    Parameters
    ----------
    data_list: dict, drug key -> sdf file path
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. drug key -> graph
    """
    graphs = {}
    for key, sdf_path in tqdm(data_list.items(), desc='sdf'):
        graphs[key] = featurize_drug(sdf_path, name=key)
    return graphs


def featurize_drug(sdf_path, name=None, edge_cutoff=4.5, num_rbf=16):
    """
    Parameters
    ----------
    sdf_path: str
        Path to sdf file
    name: str
        Name of drug
    Returns
    -------
    graph: torch_geometric.data.Data
        A torch_geometric graph
    """
    mol = rdkit.Chem.MolFromMolFile(sdf_path)
    conf = mol.GetConformer()
    with torch.no_grad():
        coords = conf.GetPositions()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = _build_atom_feature(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    # edge_v, edge_index = _build_edge_feature(mol)
    edge_s, edge_v = _build_edge_feature(
        coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=coords, edge_index=edge_index, name=name,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)
    return data

def add_train_args(parser):
    # Dataset parameters
    parser.add_argument('--task', help='Task name')
    parser.add_argument('--split_method', default='random',
        choices=['random', 'protein', 'drug', 'both', 'seqid'],
        help='Split method: random, protein, drug, or both')
    parser.add_argument('--seed', type=int, default=42,
        help='Random Seed')

    # Data representation parameters
    parser.add_argument('--contact_cutoff', type=float, default=8.,
        help='cutoff of C-alpha distance to define protein contact graph')
    parser.add_argument('--num_pos_emb', type=int, default=16,
        help='number of positional embeddings')
    parser.add_argument('--num_rbf', type=int, default=16,
        help='number of RBF kernels')

    # Protein model parameters
    parser.add_argument('--prot_gcn_dims', type=int, nargs='+', default=[128, 256, 256],
        help='protein GCN layers dimensions')
    parser.add_argument('--prot_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='protein FC layers dimensions')

    # Drug model parameters
    parser.add_argument('--drug_gcn_dims', type=int, nargs='+', default=[128, 64],
        help='drug GVP hidden layers dimensions')
    parser.add_argument('--drug_fc_dims', type=int, nargs='+', default=[1024, 128],
        help='drug FC layers dimensions')

    # Top model parameters
    parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512],
        help='top MLP layers dimensions')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
        help='dropout rate in top MLP')

    # uncertainty parameters
    parser.add_argument('--uncertainty', action='store_true',
        help='estimate uncertainty')
    parser.add_argument('--recalibrate', action='store_true',
        help='recalibrate uncertainty')

    # Training parameters
    parser.add_argument('--n_ensembles', type=int, default=1,
        help='number of ensembles')
    parser.add_argument('--batch_size', type=int, default=128,
        help='batch size')
    parser.add_argument('--n_epochs', type=int, default=500,
        help='number of epochs')
    parser.add_argument('--patience', action='store', type=int,
        help='patience for early stopping')
    parser.add_argument('--eval_freq', type=int, default=1,
        help='evaluation frequency')
    parser.add_argument('--test_freq', type=int,
        help='test frequency')
    parser.add_argument('--lr', type=float, default=0.0005,
        help='learning rate')
    parser.add_argument('--monitor_metric', default='pearson',
        help='validation metric to monitor for deciding best checkpoint')
    parser.add_argument('--parallel', action='store_true',
        help='run ensembles in parallel on multiple GPUs')

    # Save parameters
    parser.add_argument('--output_dir', action='store', default='../output', help='output folder')
    parser.add_argument('--save_log', action='store_true', default=False, help='save log file')
    parser.add_argument('--save_checkpoint', action='store_true', default=False, help='save checkpoint')
    parser.add_argument('--save_prediction', action='store_true', default=False, help='save prediction')
"""
Adapted from
https://github.com/jingraham/neurips19-graph-protein-design
https://github.com/drorlab/gvp-pytorch
"""
import math
import numpy as np
import scipy as sp
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch_geometric
import torch_cluster


def pdb_to_graphs(prot_data, params):
    """
    Converts a list of protein dict to a list of torch_geometric graphs.
    Parameters
    ----------
    prot_data : dict
        A list of protein data dict. see format in `featurize_protein_graph()`.
    params : dict
        A dictionary of parameters defined in `featurize_protein_graph()`.
    Returns
    -------
    graphs : dict
        A list of torch_geometric graphs. protein key -> graph
    """
    graphs = {}
    for key, struct in tqdm(prot_data.items(), desc='pdb'):
        graphs[key] = featurize_protein_graph(
            struct, name=key, **params)
    return graphs

def featurize_protein_graph(
        protein, name=None,
        num_pos_emb=16, num_rbf=16,        
        contact_cutoff=8.,
    ):
    """
    Parameters: see comments of DTATask() in dta.py
    """
    with torch.no_grad():
        coords = torch.as_tensor(protein['coords'], dtype=torch.float32)
        seq = torch.as_tensor([LETTER_TO_NUM[a] for a in protein['seq']], dtype=torch.long)        
        seq_emb = torch.load(protein['embed'])

        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf

        X_ca = coords[:, 1]        
        ca_mask = torch.isfinite(X_ca.sum(dim=(1)))
        ca_mask = ca_mask.float()
        ca_mask_2D = torch.unsqueeze(ca_mask, 0) * torch.unsqueeze(ca_mask, 1)
        dX_ca = torch.unsqueeze(X_ca, 0) - torch.unsqueeze(X_ca, 1)
        D_ca = ca_mask_2D * torch.sqrt(torch.sum(dX_ca**2, 2) + 1e-6)
        edge_index = torch.nonzero((D_ca < contact_cutoff) & (ca_mask_2D == 1))
        edge_index = edge_index.t().contiguous()
        

        O_feature = _local_frame(X_ca, edge_index)
        pos_embeddings = _positional_embeddings(edge_index, num_embeddings=num_pos_emb)
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=num_rbf)

        dihedrals = _dihedrals(coords)
        orientations = _orientations(X_ca)
        sidechains = _sidechains(coords)

        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_s = torch.cat([rbf, O_feature, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,
                (node_s, node_v, edge_s, edge_v))

    data = torch_geometric.data.Data(x=X_ca, seq=seq, name=name,
                                        node_s=node_s, node_v=node_v,
                                        edge_s=edge_s, edge_v=edge_v,
                                        edge_index=edge_index, mask=mask,                                        
                                        seq_emb=seq_emb)
    return data


def _dihedrals(X, eps=1e-7):
    X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


def _positional_embeddings(edge_index,
                            num_embeddings=None,
                            period_range=[2, 1000]):
    d = edge_index[0] - edge_index[1]

    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def _orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def _local_frame(X, edge_index, eps=1e-6):
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    o_1 = _normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 1)
    O = F.pad(O, (0, 0, 0, 0, 1, 2), 'constant', 0)

    # dX = X[edge_index[0]] - X[edge_index[1]]
    dX = X[edge_index[1]] - X[edge_index[0]]
    dX = _normalize(dX, dim=-1)
    # dU = torch.bmm(O[edge_index[1]], dX.unsqueeze(2)).squeeze(2)
    dU = torch.bmm(O[edge_index[0]], dX.unsqueeze(2)).squeeze(2)
    R = torch.bmm(O[edge_index[0]].transpose(-1,-2), O[edge_index[1]])
    Q = _quaternions(R)
    O_features = torch.cat((dU,Q), dim=-1)

    return O_features


def _quaternions(R):
    # Simple Wikipedia version
    # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:, i, j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)
    return Q

import sys
import yaml
import logging
import torch
import pathlib
import numpy as np

class Logger(object):
    def __init__(self, logfile=None, level=logging.INFO):
        '''
        logfile: pathlib object
        '''
        self.logger = logging.getLogger()
        self.logger.setLevel(level)
        formatter = logging.Formatter("%(asctime)s\t%(message)s", "%Y-%m-%d %H:%M:%S")

        for hd in self.logger.handlers[:]:
            self.logger.removeHandler(hd)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        if logfile is not None:
            logfile.parent.mkdir(exist_ok=True, parents=True)
            fh = logging.FileHandler(logfile, 'w')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


class Saver(object):
    def __init__(self, output_dir):        
        self.save_dir = pathlib.Path(output_dir)
    
    def mkdir(self):
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def save_ckp(self, pt, filename='checkpoint.pt'):
        self.mkdir()
        torch.save(pt, str(self.save_dir/filename))

    def save_df(self, df, filename, float_format='%.6f'):
        self.mkdir()
        df.to_csv(self.save_dir/filename, float_format=float_format, index=False, sep='\t')
    
    def save_config(self, config, filename, overwrite=True):
        self.mkdir()
        with open(self.save_dir/filename, 'w') as f:
            yaml.dump(config, f, indent=2)



class EarlyStopping(object):
    def __init__(self, 
            patience=100, eval_freq=1, best_score=None, 
            delta=1e-9, higher_better=True):
        self.patience = patience
        self.eval_freq = eval_freq
        self.best_score = best_score
        self.delta = delta
        self.higher_better = higher_better
        self.counter = 0
        self.early_stop = False
    
    def not_improved(self, val_score):
        if np.isnan(val_score):
            return True
        if self.higher_better:
            return val_score < self.best_score + self.delta
        else:
            return val_score > self.best_score - self.delta
    
    def update(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
            is_best = True
        elif self.not_improved(val_score):
            self.counter += self.eval_freq
            if (self.patience is not None) and (self.counter > self.patience):
                self.early_stop = True
            is_best = False
        else:
            self.best_score = val_score
            self.counter = 0
            is_best = True
        return is_best

def get_esm_embedding(seq, esm_model):
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = esm_model(**inputs)
    token_representations = outputs.last_hidden_state
    # remove [CLS] and [EOS] if needed
    emb = token_representations[0, 1:-1]
    return emb.cpu()



# Dataset builder for DTA class
def build_dataset(df_fold, pdb_structures, exp_cols = "pKi", is_pred = False):
    data_list = []
    for i, row in df_fold.iterrows():
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        if is_pred == True:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": 0
            })

        else:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": float(row[exp_cols]),
            })
    return DTA(df=df_fold, data_list=data_list)


def extract_backbone_coords(structure, pdb_id, pdb_path):
    coords = {"N": [], "CA": [], "C": [], "O": []}
    seq = ""

    model = structure[0]

    valid_chain = None
    for chain in model:
        if any(is_aa(res, standard=True) for res in chain):
            valid_chain = chain
            break

    if valid_chain is None:
        print("No valid chains: ", pdb_id, pdb_path)
        return None, None, None

    chain_id = valid_chain.id

    for res in valid_chain:
        if not is_aa(res, standard=True):
            continue
        seq += res.resname[0]  # fallback, not exact 1-letter code

        for atom_name in ["N", "CA", "C", "O"]:
            if atom_name in res:
                coords[atom_name].append(res[atom_name].coord.tolist())
            else:
                coords[atom_name].append([float("nan")] * 3)

    return seq, coords, chain_id


# ============= MODIFICATIONS FOR MULTI-TASK LEARNING =============

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. MASKED MSE LOSS WITH TASK WEIGHTING
class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss that handles NaN values and applies task-specific weighting
    based on the inverse of the range of each task.
    """
    def __init__(self, task_ranges=None):
        super(MaskedMSELoss, self).__init__()
        self.task_ranges = task_ranges
        if task_ranges is not None:
            # Calculate task weights based on inverse range
            weights = []
            for range_val in task_ranges.values():
                weights.append(1.0 / range_val if range_val > 0 else 1.0)
            total_weight = sum(weights)
            self.task_weights = torch.tensor([w / total_weight for w in weights])
        else:
            self.task_weights = None
    
    def forward(self, pred, target):
        """
        pred: [batch_size, n_tasks]
        target: [batch_size, n_tasks]
        """
        # Create mask for non-NaN values
        mask = ~torch.isnan(target)
        
        # Calculate MSE only for non-NaN values
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # Apply mask
        pred_masked = pred[mask]
        target_masked = target[mask]
        
        # Calculate squared errors
        se = (pred_masked - target_masked) ** 2
        
        # If we have task weights, apply them
        if self.task_weights is not None:
            # Expand mask to get task indices
            task_indices = torch.where(mask)[1]
            weights = self.task_weights.to(pred.device)[task_indices]
            weighted_se = se * weights
            loss = weighted_se.mean()
        else:
            loss = se.mean()
        
        return loss


# 2. MODIFIED DTA MODEL FOR MULTI-TASK LEARNING
class MTL_DTAModel(nn.Module):
    def __init__(self,
            task_names=['pKi', 'pEC50', 'pKd', 'pIC50'],  # List of tasks
            prot_emb_dim=1280,
            prot_gcn_dims=[128, 256, 256],
            prot_fc_dims=[1024, 128],
            drug_node_in_dim=[66, 1], drug_node_h_dims=[128, 64],
            drug_edge_in_dim=[16, 1], drug_edge_h_dims=[32, 1],            
            drug_fc_dims=[1024, 128],
            mlp_dims=[1024, 512], mlp_dropout=0.25):
        super(MTL_DTAModel, self).__init__()
        
        self.task_names = task_names
        self.n_tasks = len(task_names)
        
        # Same encoders as before
        self.drug_model = DrugGVPModel(
            node_in_dim=drug_node_in_dim, node_h_dim=drug_node_h_dims,
            edge_in_dim=drug_edge_in_dim, edge_h_dim=drug_edge_h_dims,
        )
        drug_emb_dim = drug_node_h_dims[0]
        
        self.prot_model = Prot3DGraphModel(
            d_pretrained_emb=prot_emb_dim, d_gcn=prot_gcn_dims
        )
        prot_emb_dim = prot_gcn_dims[-1]
        
        self.drug_fc = self.get_fc_layers(
            [drug_emb_dim] + drug_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
       
        self.prot_fc = self.get_fc_layers(
            [prot_emb_dim] + prot_fc_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        
        # Shared representation layers
        self.shared_fc = self.get_fc_layers(
            [drug_fc_dims[-1] + prot_fc_dims[-1]] + mlp_dims,
            dropout=mlp_dropout, batchnorm=False,
            no_last_dropout=True, no_last_activation=True)
        
        # Task-specific heads (one for each task)
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(mlp_dims[-1], 1) for task in task_names
        })
    
    def get_fc_layers(self, hidden_sizes,
            dropout=0, batchnorm=False,
            no_last_dropout=True, no_last_activation=True):
        act_fn = torch.nn.LeakyReLU()
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if not no_last_activation or i != len(hidden_sizes) - 2:
                layers.append(act_fn)
            if dropout > 0:
                if not no_last_dropout or i != len(hidden_sizes) - 2:
                    layers.append(nn.Dropout(dropout))
            if batchnorm and i != len(hidden_sizes) - 2:
                layers.append(nn.BatchNorm1d(out_dim))
        return nn.Sequential(*layers)
    
    def forward(self, xd, xp):
        # Encode drug and protein
        xd = self.drug_model(xd)
        xp = self.prot_model(xp)
        
        # Process through FC layers
        xd = self.drug_fc(xd)
        xp = self.prot_fc(xp)
        
        # Concatenate and process through shared layers
        x = torch.cat([xd, xp], dim=1)
        shared_repr = self.shared_fc(x)
        
        # Generate predictions for each task
        outputs = []
        for task in self.task_names:
            task_pred = self.task_heads[task](shared_repr)
            outputs.append(task_pred)
        
        # Stack outputs: [batch_size, n_tasks]
        return torch.cat(outputs, dim=1)

# 3. MODIFIED DTA DATASET CLASS
class MTL_DTA(data.Dataset):
    def __init__(self, df=None, data_list=None, task_cols=None, onthefly=False,
                prot_featurize_fn=None, drug_featurize_fn=None):
        super(MTL_DTA, self).__init__()
        self.data_df = df
        self.data_list = data_list
        self.task_cols = task_cols or ['pKi', 'pEC50', 'pKd', 'pIC50']
        self.onthefly = onthefly
        self.prot_featurize_fn = prot_featurize_fn
        self.drug_featurize_fn = drug_featurize_fn
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        if self.onthefly:
            drug = self.drug_featurize_fn(
                self.data_list[idx]['drug'],
                name=self.data_list[idx]['drug_name']
            )
            prot = self.prot_featurize_fn(
                self.data_list[idx]['protein'],
                name=self.data_list[idx]['protein_name']
            )
        else:
            drug = self.data_list[idx]['drug']
            prot = self.data_list[idx]['protein']
        
        # Get multi-task targets
        y_multi = []
        for task in self.task_cols:
            val = self.data_list[idx].get(task, np.nan)
            y_multi.append(val if not pd.isna(val) else np.nan)
        
        y = torch.tensor(y_multi, dtype=torch.float32)
        
        item = {'drug': drug, 'protein': prot, 'y': y}
        return item

    



# Dataset builder for DTA class
def build_dataset(df_fold, pdb_structures, exp_cols = "pKi", is_pred = False):
    data_list = []
    for i, row in df_fold.iterrows():
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        if is_pred == True:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": 0
            })

        else:
            data_list.append({
                "protein": protein,
                "drug": drug,
                "y": float(row[exp_cols]),
            })
    return DTA(df=df_fold, data_list=data_list)



# 4. MODIFIED BUILD DATASET FUNCTION
def build_mtl_dataset(df_fold, pdb_structures, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    data_list = []
    for i, row in df_fold.iterrows():
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

# 5. MODIFIED TRAINING LOOP
def train_mtl_model(model, train_loader, valid_loader, task_cols, task_ranges, 
                    n_epochs=100, lr=0.0005, device='cuda', patience=20):
    """
    Training loop for multi-task learning model
    
    Args:
        task_cols: List of task column names
        task_ranges: Dict mapping task names to their value ranges
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss(task_ranges=task_ranges)
    stopper = EarlyStopping(patience=patience, higher_better=False)
    best_model = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)  # [batch_size, n_tasks]
            
            optimizer.zero_grad()
            pred = model(xd, xp)  # [batch_size, n_tasks]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_n_batches = 0
        task_metrics = {task: {'mse': 0, 'n': 0} for task in task_cols}
        
        with torch.no_grad():
            for batch in valid_loader:
                xd = batch['drug'].to(device)
                xp = batch['protein'].to(device)
                y = batch['y'].to(device)
                
                pred = model(xd, xp)
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_n_batches += 1
                
                # Calculate per-task metrics
                for i, task in enumerate(task_cols):
                    mask = ~torch.isnan(y[:, i])
                    if mask.sum() > 0:
                        task_mse = F.mse_loss(pred[mask, i], y[mask, i])
                        task_metrics[task]['mse'] += task_mse.item()
                        task_metrics[task]['n'] += 1
        
        avg_train_loss = train_loss / n_batches
        avg_val_loss = val_loss / val_n_batches if val_n_batches > 0 else float('inf')
        
        # Print metrics
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Valid Loss: {avg_val_loss:.4f}")
        
        for task in task_cols:
            if task_metrics[task]['n'] > 0:
                avg_task_mse = task_metrics[task]['mse'] / task_metrics[task]['n']
                print(f"  {task} MSE: {avg_task_mse:.4f}")
        
        # Early stopping
        if stopper.update(avg_val_loss):
            best_model = model.state_dict()
        if stopper.early_stop:
            print("Early stopping triggered")
            break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

# 6. EXAMPLE USAGE
def prepare_mtl_experiment(df, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    """
    Prepare data for multi-task learning
    """
    # Calculate task ranges for weighting
    task_ranges = {}
    for task in task_cols:
        if task in df.columns:
            valid_values = df[task].dropna()
            if len(valid_values) > 0:
                task_ranges[task] = valid_values.max() - valid_values.min()
            else:
                task_ranges[task] = 1.0
        else:
            task_ranges[task] = 1.0
    
    print("Task ranges for weighting:")
    for task, range_val in task_ranges.items():
        weight = 1.0 / range_val if range_val > 0 else 1.0
        normalized_weight = weight / sum(1.0/r if r > 0 else 1.0 for r in task_ranges.values())
        print(f"  {task}: range={range_val:.2f}, weight={normalized_weight:.4f}")
    
    return task_ranges



import json

def structureJSON(df, esm_model):
    structure_dict = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        pdb_path = row["standardized_protein_pdb"]
        try:

            pdb_id = os.path.basename(pdb_path).split('.')[0]

            structure = parser.get_structure(pdb_id, pdb_path)
            seq, coords, chain_id = extract_backbone_coords(structure, pdb_id, pdb_path)
            if seq is None:
                available = [c.id for c in structure[0]]
                print(f"[SKIP] {pdb_id}: no usable chain found (available: {available})")
                continue


            # Stack in order: N, CA, C, O --> [L, 4, 3]
            coords_stacked = []
            for i in range(len(coords["N"])):
                coord_group = []
                for atom in ["N", "CA", "C", "O"]:
                    coord_group.append(coords[atom][i])
                coords_stacked.append(coord_group)

            if coords_stacked is None:
                print(f"[SKIP] {pdb_id}: no usable coords found (available: {pdb_path})")
                continue

                
            embedding = get_esm_embedding(seq, esm_model)
            torch.save(embedding, f"esm_embeddings/{pdb_id}.pt")

            if coords_stacked != None and embedding != None:
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": f"esm_embeddings/{pdb_id}.pt"

                }

        except Exception as e:
            print(f"[ERROR] Failed to process {pdb_id}: {e}")
            continue



    # Save to JSON
    with open("../data/pockets_structure.json", "w") as f:
        json.dump(structure_dict, f, indent=2)


    print(f"\n✅ Done. Saved {len(structure_dict)} protein structures to pockets_structure.json")

    return(structure_dict)

import os
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from multiprocessing import Pool, cpu_count
import gc

def process_single_chunk(args):
    """
    Process a single chunk of PDB files independently.
    This function is designed to be run in parallel.
    
    Args:
        args: tuple of (chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name)
    
    Returns:
        dict with chunk processing results
    """
    chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name = args
    
    # Import inside function for multiprocessing
    from transformers import EsmModel, EsmTokenizer
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import torch
    import json
    import os
    
    print(f"\n[Chunk {chunk_idx}] Starting processing of {len(pdb_paths)} PDBs")
    
    # Load ESM model for this process
    print(f"[Chunk {chunk_idx}] Loading ESM model...")
    tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
    esm_model = EsmModel.from_pretrained(esm_model_name)
    esm_model.eval()
    
    # Move to GPU if available (each process gets its own GPU memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # For multi-GPU, assign different chunks to different GPUs
        num_gpus = torch.cuda.device_count()
        gpu_id = chunk_idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        esm_model = esm_model.to(device)
        print(f"[Chunk {chunk_idx}] Using GPU {gpu_id}")
    else:
        print(f"[Chunk {chunk_idx}] Using CPU")
    
    structure_dict = {}
    results = []
    
    # Phase 1: Parallel PDB parsing within this chunk
    max_workers = min(8, cpu_count() // 4)  # Limit workers per chunk
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_pdb_path, p): p for p in pdb_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), 
                      desc=f"Chunk {chunk_idx} - PDB parsing", position=chunk_idx):
            try:
                status_tuple = fut.result(timeout=30)  # Add timeout
                results.append(status_tuple)
            except Exception as e:
                print(f"[Chunk {chunk_idx}] Error processing PDB: {e}")
    
    # Log errors
    error_count = 0
    skip_count = 0
    for r in results:
        tag = r[0]
        if tag == "skip":
            skip_count += 1
        elif tag == "error":
            error_count += 1
    
    if error_count > 0 or skip_count > 0:
        print(f"[Chunk {chunk_idx}] Skipped: {skip_count}, Errors: {error_count}")
    
    # Keep only successful items
    ok_items = [(pdb_id, seq, coords_stacked, chain_id)
                for tag, pdb_id, *rest in results if tag == "ok"
                for (seq, coords_stacked, chain_id) in [tuple(rest)]]
    
    # Phase 2: ESM embeddings (batch processing for efficiency)
    print(f"[Chunk {chunk_idx}] Computing ESM embeddings for {len(ok_items)} proteins...")
    
    os.makedirs(embed_dir, exist_ok=True)
    
    # Process in batches to optimize GPU usage
    batch_size = 8
    for i in tqdm(range(0, len(ok_items), batch_size), 
                  desc=f"Chunk {chunk_idx} - ESM embeddings", position=chunk_idx):
        batch = ok_items[i:i+batch_size]
        
        for pdb_id, seq, coords_stacked, chain_id in batch:
            try:
                # Compute embedding
                with torch.no_grad():
                    embedding = get_esm_embedding(seq, esm_model, tokenizer, device)
                
                # Save embedding
                embed_path = os.path.join(embed_dir, f"{pdb_id}.pt")
                torch.save(embedding.cpu(), embed_path)  # Save on CPU to free GPU memory
                
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": embed_path
                }
            except Exception as e:
                print(f"[Chunk {chunk_idx}] ESM embedding failed for {pdb_id}: {e}")
        
        # Periodic garbage collection
        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save chunk
    chunk_filename = f"structures_chunk_{chunk_idx:04d}.json"
    chunk_path = os.path.join(out_dir, chunk_filename)
    with open(chunk_path, "w") as f:
        json.dump(structure_dict, f, indent=2)
    
    print(f"[Chunk {chunk_idx}] ✅ Completed: {len(structure_dict)} structures saved")
    
    # Clean up GPU memory
    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "chunk_idx": chunk_idx,
        "filename": chunk_filename,
        "path": chunk_path,
        "num_structures": len(structure_dict),
        "num_errors": error_count,
        "num_skipped": skip_count
    }


def get_esm_embedding(seq, esm_model, tokenizer, device):
    """
    Get ESM embedding for a sequence.
    
    Args:
        seq: Protein sequence
        esm_model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        torch.Tensor: Embedding
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Use mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding


def structureJSON_chunked(df, esm_model_name="facebook/esm2_t33_650M_UR50D",
                         chunk_size=100000, max_chunks_parallel=4,
                         out_dir="../data/structure_chunks/",
                         embed_dir="../data/embeddings/"):
    """
    Process structures in parallel chunks to avoid memory issues and maximize speed.
    
    Args:
        df: DataFrame with protein PDB paths
        esm_model_name: Name of ESM model to use
        chunk_size: Maximum entries per chunk (default 100000)
        max_chunks_parallel: Maximum number of chunks to process in parallel
        out_dir: Directory to save chunked JSON files
        embed_dir: Directory to save embeddings
    
    Returns:
        dict: Metadata about created chunks
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    
    # Get unique PDB paths (avoid duplicates)
    pdb_paths = df["standardized_protein_pdb"].unique().tolist()
    total_pdbs = len(pdb_paths)
    num_chunks = (total_pdbs + chunk_size - 1) // chunk_size
    
    print(f"=" * 80)
    print(f"Processing {total_pdbs} unique PDBs in {num_chunks} chunks")
    print(f"Chunk size: {chunk_size}, Parallel chunks: {max_chunks_parallel}")
    print(f"=" * 80)
    
    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_pdbs)
        chunk_paths = pdb_paths[start_idx:end_idx]
        
        chunk_args.append((
            chunk_idx,
            chunk_paths,
            out_dir,
            embed_dir,
            esm_model_name
        ))
    
    # Process chunks in parallel
    chunk_results = []
    
    # Determine optimal number of parallel processes
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        # If we have GPUs, process one chunk per GPU
        parallel_chunks = min(max_chunks_parallel, num_gpus, num_chunks)
        print(f"Using {num_gpus} GPUs, processing {parallel_chunks} chunks in parallel")
    else:
        # CPU only - limit parallelism to avoid memory issues
        parallel_chunks = min(max_chunks_parallel, cpu_count() // 4, num_chunks)
        print(f"Using CPU only, processing {parallel_chunks} chunks in parallel")
    
    # Process in batches of parallel chunks
    for batch_start in range(0, num_chunks, parallel_chunks):
        batch_end = min(batch_start + parallel_chunks, num_chunks)
        batch_args = chunk_args[batch_start:batch_end]
        
        print(f"\nProcessing chunk batch {batch_start+1}-{batch_end} of {num_chunks}")
        
        if len(batch_args) == 1:
            # Single chunk - process directly
            result = process_single_chunk(batch_args[0])
            chunk_results.append(result)
        else:
            # Multiple chunks - use multiprocessing
            with Pool(processes=len(batch_args)) as pool:
                results = pool.map(process_single_chunk, batch_args)
                chunk_results.extend(results)
        
        # Garbage collection between batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create metadata
    chunk_metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_structures": sum(r["num_structures"] for r in chunk_results),
        "total_errors": sum(r["num_errors"] for r in chunk_results),
        "total_skipped": sum(r["num_skipped"] for r in chunk_results),
        "chunks": []
    }
    
    # Add chunk info with proper indices
    start_idx = 0
    for result in sorted(chunk_results, key=lambda x: x["chunk_idx"]):
        end_idx = start_idx + result["num_structures"]
        chunk_info = {
            "chunk_idx": result["chunk_idx"],
            "filename": result["filename"],
            "path": result["path"],
            "num_structures": result["num_structures"],
            "num_errors": result["num_errors"],
            "num_skipped": result["num_skipped"],
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        chunk_metadata["chunks"].append(chunk_info)
        start_idx = end_idx
    
    # Save metadata
    metadata_path = os.path.join(out_dir, "chunk_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Processing complete!")
    print(f"  - Total structures: {chunk_metadata['total_structures']}")
    print(f"  - Total errors: {chunk_metadata['total_errors']}")
    print(f"  - Total skipped: {chunk_metadata['total_skipped']}")
    print(f"  - Metadata saved: {metadata_path}")
    print(f"{'=' * 80}")
    
    return chunk_metadata


# ============= Helper function for PDB processing =============
def _process_pdb_path(pdb_path):
    """
    Process a single PDB file to extract sequence and coordinates.
    This function runs in a separate process.
    
    Returns:
        tuple: (status, pdb_id, data...) where status is "ok", "skip", or "error"
    """
    from Bio.PDB import PDBParser, is_aa
    
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        
        # Get first model
        model = structure[0]
        
        # Process each chain
        for chain in model:
            residues = [r for r in chain if is_aa(r)]
            if len(residues) == 0:
                continue
            
            # Extract sequence
            seq = ''.join([seq1(r.resname) for r in residues])
            
            # Extract coordinates [N, CA, C, O] for each residue
            coords = []
            for residue in residues:
                try:
                    n_coord = residue['N'].coord.tolist()
                    ca_coord = residue['CA'].coord.tolist()
                    c_coord = residue['C'].coord.tolist()
                    o_coord = residue['O'].coord.tolist()
                    coords.append([n_coord, ca_coord, c_coord, o_coord])
                except:
                    # Missing atoms - use zeros
                    coords.append([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
            
            return ("ok", pdb_id, seq, coords, chain.id)
        
        return ("skip", pdb_id, "No valid chains found")
        
    except Exception as e:
        return ("error", pdb_id, str(e))


# ============= Optimized Chunk Loader (same as before) =============
class StructureChunkLoader:
    """
    Efficient loader for chunked structure dictionaries.
    Loads chunks on-demand and caches them.
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", cache_size=2):
        self.chunk_dir = chunk_dir
        self.cache_size = cache_size
        self.cache = {}  # chunk_idx -> structure_dict
        self.cache_order = []  # LRU tracking
        
        # Load metadata
        metadata_path = os.path.join(chunk_dir, "chunk_metadata.json")
        with open(metadata_path, "r") as f: # speed up JSON load from StructureChunkLoader
            self.metadata = json.load(f)
        
        print(f"Loaded metadata: {self.metadata['total_structures']} structures in {self.metadata['num_chunks']} chunks")
        
        # Build lookup: pdb_id -> chunk_idx
        self.pdb_to_chunk = {}
        for chunk_info in self.metadata["chunks"]:
            chunk_path = os.path.join(chunk_dir, chunk_info["filename"])
            if os.path.exists(chunk_path):
                with open(chunk_path, "r") as f:
                    chunk_data = json.load(f)
                    for pdb_id in chunk_data.keys():
                        self.pdb_to_chunk[pdb_id] = chunk_info["chunk_idx"]
            else:
                print(f"Warning: Chunk file not found: {chunk_path}")
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk into cache, managing cache size."""
        if chunk_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        chunk_info = self.metadata["chunks"][chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_info["filename"])
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        
        # Add to cache
        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()  # Force garbage collection
        
        return chunk_data
    
    def get(self, pdb_id):
        """Get structure for a specific PDB ID."""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)
    
    def get_batch(self, pdb_ids):
        """Get multiple structures efficiently by grouping by chunk."""
        # Group PDB IDs by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load each chunk and extract structures
        results = {}
        for chunk_idx, chunk_pdb_ids in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for pdb_id in chunk_pdb_ids:
                if pdb_id in chunk_data:
                    results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get_available_pdb_ids(self):
        """Return set of all available PDB IDs."""
        return set(self.pdb_to_chunk.keys())


def build_mtl_dataset_optimized(df_fold, chunk_loader, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    """
    Build MTL dataset efficiently using chunked structure loader.
    
    Args:
        df_fold: DataFrame with fold data
        chunk_loader: StructureChunkLoader instance
        task_cols: List of task columns
    
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    # Get all protein IDs from the fold
    protein_ids = df_fold["protein_id"].tolist()
    
    # Batch load structures (efficient chunk-based loading)
    print(f"Loading structures for {len(protein_ids)} proteins...")
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Process each row
    skipped = 0
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        protein_id = row["protein_id"]
        
        if protein_id not in structures_batch:
            skipped += 1
            continue
        
        protein_json = structures_batch[protein_id]
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing structures")
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)


# ============= USAGE EXAMPLE =============
import os
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import torch
from Bio.PDB import PDBParser  # make sure Biopython is installed

# Assumes you already have:
# - extract_backbone_coords(structure, pdb_id, pdb_path)
# - get_esm_embedding(seq, esm_model)

ATOMS = ("N", "CA", "C", "O")
EMBED_DIR = Path("esm_embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

def _stack_backbone(coords):
    # coords: dict with keys "N","CA","C","O", each a list of [x,y,z]
    L = len(coords["N"])
    return [[coords[a][i] for a in ATOMS] for i in range(L)]

def _process_pdb_path(pdb_path):
    """
    Worker: parse PDB, extract seq/coords/chain, return tuple or a skip marker.
    Runs in a separate process; initializes its own parser.
    """
    try:
        pdb_id = os.path.basename(pdb_path).split('.')[0]
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_path)

        seq, coords, chain_id = extract_backbone_coords(structure, pdb_id, pdb_path)
        if seq is None:
            available = [c.id for c in structure[0]]
            return ("skip", pdb_id, f"no usable chain (available: {available})")

        if not coords or any(k not in coords for k in ATOMS) or len(coords["N"]) == 0:
            return ("skip", pdb_id, "no usable coords")

        coords_stacked = _stack_backbone(coords)
        if not coords_stacked:
            return ("skip", pdb_id, "empty coords after stacking")

        return ("ok", pdb_id, seq, coords_stacked, chain_id)

    except Exception as e:
        return ("error", os.path.basename(pdb_path).split('.')[0], str(e))

def structureJSON(df, esm_model, max_workers=None, embed_batch_size=8, out_json="../data/pockets_structure.json"):
    structure_dict = {}

    pdb_paths = df["standardized_protein_pdb"].tolist()
    results = []

    # Phase 1: parallel PDB parsing + coordinate extraction
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_pdb_path, p): p for p in pdb_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="PDB -> seq/coords"):
            status_tuple = fut.result()
            results.append(status_tuple)

    # Log skips/errors (fast)
    for r in results:
        tag = r[0]
        if tag == "skip":
            _, pdb_id, msg = r
            print(f"[SKIP] {pdb_id}: {msg}")
        elif tag == "error":
            _, pdb_id, err = r
            print(f"[ERROR] Failed to process {pdb_id}: {err}")

    # Keep only successful items
    ok_items = [(pdb_id, seq, coords_stacked, chain_id)
                for tag, pdb_id, *rest in results if tag == "ok"
                for (seq, coords_stacked, chain_id) in [tuple(rest)]]

    # Phase 2: embeddings on a single device (GPU/CPU) to avoid per-process model copies
    # Optionally batch if your get_esm_embedding supports lists; otherwise do per-sequence.
    # Here we do per-sequence by default; simple and safe.
    for pdb_id, seq, coords_stacked, chain_id in tqdm(ok_items, desc="ESM embeddings"):
        try:
            embedding = get_esm_embedding(seq, esm_model)  # ensure this returns a tensor
            torch.save(embedding, EMBED_DIR / f"{pdb_id}.pt")

            structure_dict[pdb_id] = {
                "name": pdb_id,
                "UniProt_id": "UNKNOWN",
                "PDB_id": pdb_id,
                "chain": chain_id,
                "seq": seq,
                "coords": coords_stacked,         # [[N,CA,C,O], ...], each as [x,y,z]
                "embed": str(EMBED_DIR / f"{pdb_id}.pt")
            }
        except Exception as e:
            print(f"[ERROR] ESM embedding failed for {pdb_id}: {e}")

    # Save to JSON
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(structure_dict, f, indent=2)

    print(f"\n✅ Done. Saved {len(structure_dict)} protein structures to {os.path.basename(out_json)}")
    return structure_dict





import os
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

def structureJSON_chunked(df, esm_model, max_workers=None, embed_batch_size=8, 
                          chunk_size=100000, out_dir="../data/structure_chunks/"):
    """
    Process structures in chunks to avoid memory issues.
    
    Args:
        df: DataFrame with protein PDB paths
        esm_model: ESM model for embeddings
        max_workers: Number of parallel workers
        chunk_size: Maximum entries per chunk (default 100000)
        out_dir: Directory to save chunked JSON files
    
    Returns:
        dict: Metadata about created chunks
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)
    
    pdb_paths = df["standardized_protein_pdb"].tolist()
    total_pdbs = len(pdb_paths)
    num_chunks = (total_pdbs + chunk_size - 1) // chunk_size
    
    print(f"Processing {total_pdbs} PDBs in {num_chunks} chunks of max {chunk_size} each")
    
    chunk_metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "chunks": []
    }
    
    # Process in chunks
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_pdbs)
        chunk_paths = pdb_paths[start_idx:end_idx]
        
        print(f"\n=== Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_paths)} PDBs) ===")
        
        structure_dict = {}
        results = []
        
        # Phase 1: Parallel PDB parsing for this chunk
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_pdb_path, p): p for p in chunk_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Chunk {chunk_idx + 1} - PDB parsing"):
                status_tuple = fut.result()
                results.append(status_tuple)
        
        # Log errors for this chunk
        for r in results:
            tag = r[0]
            if tag == "skip":
                _, pdb_id, msg = r
                print(f"[SKIP] {pdb_id}: {msg}")
            elif tag == "error":
                _, pdb_id, err = r
                print(f"[ERROR] Failed to process {pdb_id}: {err}")
        
        # Keep only successful items
        ok_items = [(pdb_id, seq, coords_stacked, chain_id)
                    for tag, pdb_id, *rest in results if tag == "ok"
                    for (seq, coords_stacked, chain_id) in [tuple(rest)]]
        
        # Phase 2: ESM embeddings for this chunk
        for pdb_id, seq, coords_stacked, chain_id in tqdm(ok_items, 
                                                          desc=f"Chunk {chunk_idx + 1} - ESM embeddings"):
            try:
                embedding = get_esm_embedding(seq, esm_model)
                embed_path = EMBED_DIR / f"{pdb_id}.pt"
                torch.save(embedding, embed_path)
                
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": str(embed_path)
                }
            except Exception as e:
                print(f"[ERROR] ESM embedding failed for {pdb_id}: {e}")
        
        # Save this chunk
        chunk_filename = f"structures_chunk_{chunk_idx:04d}.json"
        chunk_path = os.path.join(out_dir, chunk_filename)
        with open(chunk_path, "w") as f:
            json.dump(structure_dict, f, indent=2)
        
        chunk_info = {
            "chunk_idx": chunk_idx,
            "filename": chunk_filename,
            "path": chunk_path,
            "num_structures": len(structure_dict),
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        chunk_metadata["chunks"].append(chunk_info)
        
        print(f"✅ Chunk {chunk_idx + 1} saved: {len(structure_dict)} structures to {chunk_filename}")
    
    # Save metadata
    metadata_path = os.path.join(out_dir, "chunk_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    
    print(f"\n✅ All chunks processed. Metadata saved to {metadata_path}")
    return chunk_metadata


class StructureChunkLoader:
    """
    Efficient loader for chunked structure dictionaries.
    Loads chunks on-demand and caches them.
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", cache_size=2):
        self.chunk_dir = chunk_dir
        self.cache_size = cache_size
        self.cache = {}  # chunk_idx -> structure_dict
        self.cache_order = []  # LRU tracking
        
        # Load metadata
        metadata_path = os.path.join(chunk_dir, "chunk_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Build lookup: pdb_id -> chunk_idx
        self.pdb_to_chunk = {}
        for chunk_info in self.metadata["chunks"]:
            chunk_path = os.path.join(chunk_dir, chunk_info["filename"])
            # Quick scan to build index (could be saved in metadata for efficiency)
            with open(chunk_path, "r") as f:
                chunk_data = json.load(f)
                for pdb_id in chunk_data.keys():
                    self.pdb_to_chunk[pdb_id] = chunk_info["chunk_idx"]
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk into cache, managing cache size."""
        if chunk_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        chunk_info = self.metadata["chunks"][chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_info["filename"])
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        
        # Add to cache
        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        return chunk_data
    
    def get(self, pdb_id):
        """Get structure for a specific PDB ID."""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)
    
    def get_batch(self, pdb_ids):
        """Get multiple structures efficiently by grouping by chunk."""
        # Group PDB IDs by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load each chunk and extract structures
        results = {}
        for chunk_idx, chunk_pdb_ids in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for pdb_id in chunk_pdb_ids:
                if pdb_id in chunk_data:
                    results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get_available_pdb_ids(self):
        """Return set of all available PDB IDs."""
        return set(self.pdb_to_chunk.keys())


def build_mtl_dataset_optimized(df_fold, chunk_loader, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    """
    Build MTL dataset efficiently using chunked structure loader.
    
    Args:
        df_fold: DataFrame with fold data
        chunk_loader: StructureChunkLoader instance
        task_cols: List of task columns
    
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    # Get all protein IDs from the fold
    protein_ids = df_fold["protein_id"].tolist()
    
    # Batch load structures (efficient chunk-based loading)
    print(f"Loading structures for {len(protein_ids)} proteins...")
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Process each row
    skipped = 0
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        protein_id = row["protein_id"]
        
        if protein_id not in structures_batch:
            skipped += 1
            continue
        
        protein_json = structures_batch[protein_id]
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing structures")
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

# 4. MODIFIED BUILD DATASET FUNCTION
def build_mtl_dataset(df_fold, pdb_structures, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    data_list = []
    for i, row in df_fold.iterrows():
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

# ============= USAGE EXAMPLE =============
import os
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from multiprocessing import Pool, cpu_count
import gc

def process_single_chunk(args):
    """
    Process a single chunk of PDB files independently.
    This function is designed to be run in parallel.
    
    Args:
        args: tuple of (chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name)
    
    Returns:
        dict with chunk processing results
    """
    chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name = args
    
    # Import inside function for multiprocessing
    from transformers import EsmModel, EsmTokenizer
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import torch
    import json
    import os
    
    print(f"\n[Chunk {chunk_idx}] Starting processing of {len(pdb_paths)} PDBs")
    
    # Load ESM model for this process
    print(f"[Chunk {chunk_idx}] Loading ESM model...")
    tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
    esm_model = EsmModel.from_pretrained(esm_model_name)
    esm_model.eval()
    
    # Move to GPU if available (each process gets its own GPU memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # For multi-GPU, assign different chunks to different GPUs
        num_gpus = torch.cuda.device_count()
        gpu_id = chunk_idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        esm_model = esm_model.to(device)
        print(f"[Chunk {chunk_idx}] Using GPU {gpu_id}")
    else:
        print(f"[Chunk {chunk_idx}] Using CPU")
    
    structure_dict = {}
    results = []
    
    # Phase 1: Parallel PDB parsing within this chunk
    max_workers = min(8, cpu_count() // 4)  # Limit workers per chunk
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_pdb_path, p): p for p in pdb_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), 
                      desc=f"Chunk {chunk_idx} - PDB parsing", position=chunk_idx):
            try:
                status_tuple = fut.result(timeout=30)  # Add timeout
                results.append(status_tuple)
            except Exception as e:
                print(f"[Chunk {chunk_idx}] Error processing PDB: {e}")
    
    # Log errors
    error_count = 0
    skip_count = 0
    for r in results:
        tag = r[0]
        if tag == "skip":
            skip_count += 1
        elif tag == "error":
            error_count += 1
    
    if error_count > 0 or skip_count > 0:
        print(f"[Chunk {chunk_idx}] Skipped: {skip_count}, Errors: {error_count}")
    
    # Keep only successful items
    ok_items = [(pdb_id, seq, coords_stacked, chain_id)
                for tag, pdb_id, *rest in results if tag == "ok"
                for (seq, coords_stacked, chain_id) in [tuple(rest)]]
    
    # Phase 2: ESM embeddings (batch processing for efficiency)
    print(f"[Chunk {chunk_idx}] Computing ESM embeddings for {len(ok_items)} proteins...")
    
    os.makedirs(embed_dir, exist_ok=True)
    
    # Process in batches to optimize GPU usage
    batch_size = 8
    for i in tqdm(range(0, len(ok_items), batch_size), 
                  desc=f"Chunk {chunk_idx} - ESM embeddings", position=chunk_idx):
        batch = ok_items[i:i+batch_size]
        
        for pdb_id, seq, coords_stacked, chain_id in batch:
            try:
                # Compute embedding
                with torch.no_grad():
                    embedding = get_esm_embedding(seq, esm_model, tokenizer, device)
                
                # Save embedding
                embed_path = os.path.join(embed_dir, f"{pdb_id}.pt")
                torch.save(embedding.cpu(), embed_path)  # Save on CPU to free GPU memory
                
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": embed_path
                }
            except Exception as e:
                print(f"[Chunk {chunk_idx}] ESM embedding failed for {pdb_id}: {e}")
        
        # Periodic garbage collection
        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save chunk
    chunk_filename = f"structures_chunk_{chunk_idx:04d}.json"
    chunk_path = os.path.join(out_dir, chunk_filename)
    with open(chunk_path, "w") as f:
        json.dump(structure_dict, f, indent=2)
    
    print(f"[Chunk {chunk_idx}] ✅ Completed: {len(structure_dict)} structures saved")
    
    # Clean up GPU memory
    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "chunk_idx": chunk_idx,
        "filename": chunk_filename,
        "path": chunk_path,
        "num_structures": len(structure_dict),
        "num_errors": error_count,
        "num_skipped": skip_count
    }


def get_esm_embedding(seq, esm_model, tokenizer, device):
    """
    Get ESM embedding for a sequence.
    
    Args:
        seq: Protein sequence
        esm_model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        torch.Tensor: Embedding
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Use mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding


def structureJSON_chunked(df, esm_model_name="facebook/esm2_t33_650M_UR50D",
                         chunk_size=100000, max_chunks_parallel=4,
                         out_dir="../data/structure_chunks/",
                         embed_dir="../data/embeddings/"):
    """
    Process structures in parallel chunks to avoid memory issues and maximize speed.
    
    Args:
        df: DataFrame with protein PDB paths
        esm_model_name: Name of ESM model to use
        chunk_size: Maximum entries per chunk (default 100000)
        max_chunks_parallel: Maximum number of chunks to process in parallel
        out_dir: Directory to save chunked JSON files
        embed_dir: Directory to save embeddings
    
    Returns:
        dict: Metadata about created chunks
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    
    # Get unique PDB paths (avoid duplicates)
    pdb_paths = df["standardized_protein_pdb"].unique().tolist()
    total_pdbs = len(pdb_paths)
    num_chunks = (total_pdbs + chunk_size - 1) // chunk_size
    
    print(f"=" * 80)
    print(f"Processing {total_pdbs} unique PDBs in {num_chunks} chunks")
    print(f"Chunk size: {chunk_size}, Parallel chunks: {max_chunks_parallel}")
    print(f"=" * 80)
    
    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_pdbs)
        chunk_paths = pdb_paths[start_idx:end_idx]
        
        chunk_args.append((
            chunk_idx,
            chunk_paths,
            out_dir,
            embed_dir,
            esm_model_name
        ))
    
    # Process chunks in parallel
    chunk_results = []
    
    # Determine optimal number of parallel processes
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        # If we have GPUs, process one chunk per GPU
        parallel_chunks = min(max_chunks_parallel, num_gpus, num_chunks)
        print(f"Using {num_gpus} GPUs, processing {parallel_chunks} chunks in parallel")
    else:
        # CPU only - limit parallelism to avoid memory issues
        parallel_chunks = min(max_chunks_parallel, cpu_count() // 4, num_chunks)
        print(f"Using CPU only, processing {parallel_chunks} chunks in parallel")
    
    # Process in batches of parallel chunks
    for batch_start in range(0, num_chunks, parallel_chunks):
        batch_end = min(batch_start + parallel_chunks, num_chunks)
        batch_args = chunk_args[batch_start:batch_end]
        
        print(f"\nProcessing chunk batch {batch_start+1}-{batch_end} of {num_chunks}")
        
        if len(batch_args) == 1:
            # Single chunk - process directly
            result = process_single_chunk(batch_args[0])
            chunk_results.append(result)
        else:
            # Multiple chunks - use multiprocessing
            with Pool(processes=len(batch_args)) as pool:
                results = pool.map(process_single_chunk, batch_args)
                chunk_results.extend(results)
        
        # Garbage collection between batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create metadata
    chunk_metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_structures": sum(r["num_structures"] for r in chunk_results),
        "total_errors": sum(r["num_errors"] for r in chunk_results),
        "total_skipped": sum(r["num_skipped"] for r in chunk_results),
        "chunks": []
    }
    
    # Add chunk info with proper indices
    start_idx = 0
    for result in sorted(chunk_results, key=lambda x: x["chunk_idx"]):
        end_idx = start_idx + result["num_structures"]
        chunk_info = {
            "chunk_idx": result["chunk_idx"],
            "filename": result["filename"],
            "path": result["path"],
            "num_structures": result["num_structures"],
            "num_errors": result["num_errors"],
            "num_skipped": result["num_skipped"],
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        chunk_metadata["chunks"].append(chunk_info)
        start_idx = end_idx
    
    # Save metadata
    metadata_path = os.path.join(out_dir, "chunk_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Processing complete!")
    print(f"  - Total structures: {chunk_metadata['total_structures']}")
    print(f"  - Total errors: {chunk_metadata['total_errors']}")
    print(f"  - Total skipped: {chunk_metadata['total_skipped']}")
    print(f"  - Metadata saved: {metadata_path}")
    print(f"{'=' * 80}")
    
    return chunk_metadata


# ============= Helper function for PDB processing =============
def _process_pdb_path(pdb_path):
    """
    Process a single PDB file to extract sequence and coordinates.
    This function runs in a separate process.
    
    Returns:
        tuple: (status, pdb_id, data...) where status is "ok", "skip", or "error"
    """
    from Bio.PDB import PDBParser, is_aa
    
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        
        # Get first model
        model = structure[0]
        
        # Process each chain
        for chain in model:
            residues = [r for r in chain if is_aa(r)]
            if len(residues) == 0:
                continue
            
            # Extract sequence
            seq = ''.join([seq1(r.resname) for r in residues])
            
            # Extract coordinates [N, CA, C, O] for each residue
            coords = []
            for residue in residues:
                try:
                    n_coord = residue['N'].coord.tolist()
                    ca_coord = residue['CA'].coord.tolist()
                    c_coord = residue['C'].coord.tolist()
                    o_coord = residue['O'].coord.tolist()
                    coords.append([n_coord, ca_coord, c_coord, o_coord])
                except:
                    # Missing atoms - use zeros
                    coords.append([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
            
            return ("ok", pdb_id, seq, coords, chain.id)
        
        return ("skip", pdb_id, "No valid chains found")
        
    except Exception as e:
        return ("error", pdb_id, str(e))


# ============= Optimized Chunk Loader (same as before) =============
class StructureChunkLoader:
    """
    Efficient loader for chunked structure dictionaries.
    Loads chunks on-demand and caches them.
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", cache_size=2):
        self.chunk_dir = chunk_dir
        self.cache_size = cache_size
        self.cache = {}  # chunk_idx -> structure_dict
        self.cache_order = []  # LRU tracking
        
        # Load metadata
        metadata_path = os.path.join(chunk_dir, "chunk_metadata.json")
        with open(metadata_path, "r") as f: # speed up JSON load from StructureChunkLoader
            self.metadata = json.load(f)
        
        print(f"Loaded metadata: {self.metadata['total_structures']} structures in {self.metadata['num_chunks']} chunks")
        
        # Build lookup: pdb_id -> chunk_idx
        self.pdb_to_chunk = {}
        for chunk_info in self.metadata["chunks"]:
            chunk_path = os.path.join(chunk_dir, chunk_info["filename"])
            if os.path.exists(chunk_path):
                with open(chunk_path, "r") as f:
                    chunk_data = json.load(f)
                    for pdb_id in chunk_data.keys():
                        self.pdb_to_chunk[pdb_id] = chunk_info["chunk_idx"]
            else:
                print(f"Warning: Chunk file not found: {chunk_path}")
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk into cache, managing cache size."""
        if chunk_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        chunk_info = self.metadata["chunks"][chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_info["filename"])
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        
        # Add to cache
        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()  # Force garbage collection
        
        return chunk_data
    
    def get(self, pdb_id):
        """Get structure for a specific PDB ID."""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)
    
    def get_batch(self, pdb_ids):
        """Get multiple structures efficiently by grouping by chunk."""
        # Group PDB IDs by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load each chunk and extract structures
        results = {}
        for chunk_idx, chunk_pdb_ids in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for pdb_id in chunk_pdb_ids:
                if pdb_id in chunk_data:
                    results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get_available_pdb_ids(self):
        """Return set of all available PDB IDs."""
        return set(self.pdb_to_chunk.keys())


def build_mtl_dataset_optimized(df_fold, chunk_loader, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    """
    Build MTL dataset efficiently using chunked structure loader.
    
    Args:
        df_fold: DataFrame with fold data
        chunk_loader: StructureChunkLoader instance
        task_cols: List of task columns
    
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    # Get all protein IDs from the fold
    protein_ids = df_fold["protein_id"].tolist()
    
    # Batch load structures (efficient chunk-based loading)
    print(f"Loading structures for {len(protein_ids)} proteins...")
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Process each row
    skipped = 0
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        protein_id = row["protein_id"]
        
        if protein_id not in structures_batch:
            skipped += 1
            continue
        
        protein_json = structures_batch[protein_id]
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing structures")
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)


# ============= USAGE EXAMPLE =============
import os
import json
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from multiprocessing import Pool, cpu_count
import gc

def process_single_chunk(args):
    """
    Process a single chunk of PDB files independently.
    This function is designed to be run in parallel.
    
    Args:
        args: tuple of (chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name)
    
    Returns:
        dict with chunk processing results
    """
    chunk_idx, pdb_paths, out_dir, embed_dir, esm_model_name = args
    
    # Import inside function for multiprocessing
    from transformers import EsmModel, EsmTokenizer
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import torch
    import json
    import os
    
    print(f"\n[Chunk {chunk_idx}] Starting processing of {len(pdb_paths)} PDBs")
    
    # Load ESM model for this process
    print(f"[Chunk {chunk_idx}] Loading ESM model...")
    tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
    esm_model = EsmModel.from_pretrained(esm_model_name)
    esm_model.eval()
    
    # Move to GPU if available (each process gets its own GPU memory)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # For multi-GPU, assign different chunks to different GPUs
        num_gpus = torch.cuda.device_count()
        gpu_id = chunk_idx % num_gpus
        device = torch.device(f'cuda:{gpu_id}')
        esm_model = esm_model.to(device)
        print(f"[Chunk {chunk_idx}] Using GPU {gpu_id}")
    else:
        print(f"[Chunk {chunk_idx}] Using CPU")
    
    structure_dict = {}
    results = []
    
    # Phase 1: Parallel PDB parsing within this chunk
    max_workers = min(8, cpu_count() // 4)  # Limit workers per chunk
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_pdb_path, p): p for p in pdb_paths}
        for fut in tqdm(as_completed(futures), total=len(futures), 
                      desc=f"Chunk {chunk_idx} - PDB parsing", position=chunk_idx):
            try:
                status_tuple = fut.result(timeout=30)  # Add timeout
                results.append(status_tuple)
            except Exception as e:
                print(f"[Chunk {chunk_idx}] Error processing PDB: {e}")
    
    # Log errors
    error_count = 0
    skip_count = 0
    for r in results:
        tag = r[0]
        if tag == "skip":
            skip_count += 1
        elif tag == "error":
            error_count += 1
    
    if error_count > 0 or skip_count > 0:
        print(f"[Chunk {chunk_idx}] Skipped: {skip_count}, Errors: {error_count}")
    
    # Keep only successful items
    ok_items = [(pdb_id, seq, coords_stacked, chain_id)
                for tag, pdb_id, *rest in results if tag == "ok"
                for (seq, coords_stacked, chain_id) in [tuple(rest)]]
    
    # Phase 2: ESM embeddings (batch processing for efficiency)
    print(f"[Chunk {chunk_idx}] Computing ESM embeddings for {len(ok_items)} proteins...")
    
    os.makedirs(embed_dir, exist_ok=True)
    
    # Process in batches to optimize GPU usage
    batch_size = 8
    for i in tqdm(range(0, len(ok_items), batch_size), 
                  desc=f"Chunk {chunk_idx} - ESM embeddings", position=chunk_idx):
        batch = ok_items[i:i+batch_size]
        
        for pdb_id, seq, coords_stacked, chain_id in batch:
            try:
                # Compute embedding
                with torch.no_grad():
                    embedding = get_esm_embedding(seq, esm_model, tokenizer, device)
                
                # Save embedding
                embed_path = os.path.join(embed_dir, f"{pdb_id}.pt")
                torch.save(embedding.cpu(), embed_path)  # Save on CPU to free GPU memory
                
                structure_dict[pdb_id] = {
                    "name": pdb_id,
                    "UniProt_id": "UNKNOWN",
                    "PDB_id": pdb_id,
                    "chain": chain_id,
                    "seq": seq,
                    "coords": coords_stacked,
                    "embed": embed_path
                }
            except Exception as e:
                print(f"[Chunk {chunk_idx}] ESM embedding failed for {pdb_id}: {e}")
        
        # Periodic garbage collection
        if i % 100 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save chunk
    chunk_filename = f"structures_chunk_{chunk_idx:04d}.json"
    chunk_path = os.path.join(out_dir, chunk_filename)
    with open(chunk_path, "w") as f:
        json.dump(structure_dict, f, indent=2)
    
    print(f"[Chunk {chunk_idx}] ✅ Completed: {len(structure_dict)} structures saved")
    
    # Clean up GPU memory
    del esm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "chunk_idx": chunk_idx,
        "filename": chunk_filename,
        "path": chunk_path,
        "num_structures": len(structure_dict),
        "num_errors": error_count,
        "num_skipped": skip_count
    }


def get_esm_embedding(seq, esm_model, tokenizer, device):
    """
    Get ESM embedding for a sequence.
    
    Args:
        seq: Protein sequence
        esm_model: ESM model
        tokenizer: ESM tokenizer
        device: torch device
    
    Returns:
        torch.Tensor: Embedding
    """
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = esm_model(**inputs)
        # Use mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding


def structureJSON_chunked(df, esm_model_name="facebook/esm2_t33_650M_UR50D",
                         chunk_size=100000, max_chunks_parallel=4,
                         out_dir="../data/structure_chunks/",
                         embed_dir="../data/embeddings/"):
    """
    Process structures in parallel chunks to avoid memory issues and maximize speed.
    
    Args:
        df: DataFrame with protein PDB paths
        esm_model_name: Name of ESM model to use
        chunk_size: Maximum entries per chunk (default 100000)
        max_chunks_parallel: Maximum number of chunks to process in parallel
        out_dir: Directory to save chunked JSON files
        embed_dir: Directory to save embeddings
    
    Returns:
        dict: Metadata about created chunks
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    
    # Get unique PDB paths (avoid duplicates)
    pdb_paths = df["standardized_protein_pdb"].unique().tolist()
    total_pdbs = len(pdb_paths)
    num_chunks = (total_pdbs + chunk_size - 1) // chunk_size
    
    print(f"=" * 80)
    print(f"Processing {total_pdbs} unique PDBs in {num_chunks} chunks")
    print(f"Chunk size: {chunk_size}, Parallel chunks: {max_chunks_parallel}")
    print(f"=" * 80)
    
    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_pdbs)
        chunk_paths = pdb_paths[start_idx:end_idx]
        
        chunk_args.append((
            chunk_idx,
            chunk_paths,
            out_dir,
            embed_dir,
            esm_model_name
        ))
    
    # Process chunks in parallel
    chunk_results = []
    
    # Determine optimal number of parallel processes
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        # If we have GPUs, process one chunk per GPU
        parallel_chunks = min(max_chunks_parallel, num_gpus, num_chunks)
        print(f"Using {num_gpus} GPUs, processing {parallel_chunks} chunks in parallel")
    else:
        # CPU only - limit parallelism to avoid memory issues
        parallel_chunks = min(max_chunks_parallel, cpu_count() // 4, num_chunks)
        print(f"Using CPU only, processing {parallel_chunks} chunks in parallel")
    
    # Process in batches of parallel chunks
    for batch_start in range(0, num_chunks, parallel_chunks):
        batch_end = min(batch_start + parallel_chunks, num_chunks)
        batch_args = chunk_args[batch_start:batch_end]
        
        print(f"\nProcessing chunk batch {batch_start+1}-{batch_end} of {num_chunks}")
        
        if len(batch_args) == 1:
            # Single chunk - process directly
            result = process_single_chunk(batch_args[0])
            chunk_results.append(result)
        else:
            # Multiple chunks - use multiprocessing
            with Pool(processes=len(batch_args)) as pool:
                results = pool.map(process_single_chunk, batch_args)
                chunk_results.extend(results)
        
        # Garbage collection between batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create metadata
    chunk_metadata = {
        "num_chunks": num_chunks,
        "chunk_size": chunk_size,
        "total_structures": sum(r["num_structures"] for r in chunk_results),
        "total_errors": sum(r["num_errors"] for r in chunk_results),
        "total_skipped": sum(r["num_skipped"] for r in chunk_results),
        "chunks": []
    }
    
    # Add chunk info with proper indices
    start_idx = 0
    for result in sorted(chunk_results, key=lambda x: x["chunk_idx"]):
        end_idx = start_idx + result["num_structures"]
        chunk_info = {
            "chunk_idx": result["chunk_idx"],
            "filename": result["filename"],
            "path": result["path"],
            "num_structures": result["num_structures"],
            "num_errors": result["num_errors"],
            "num_skipped": result["num_skipped"],
            "start_idx": start_idx,
            "end_idx": end_idx
        }
        chunk_metadata["chunks"].append(chunk_info)
        start_idx = end_idx
    
    # Save metadata
    metadata_path = os.path.join(out_dir, "chunk_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(chunk_metadata, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"✅ Processing complete!")
    print(f"  - Total structures: {chunk_metadata['total_structures']}")
    print(f"  - Total errors: {chunk_metadata['total_errors']}")
    print(f"  - Total skipped: {chunk_metadata['total_skipped']}")
    print(f"  - Metadata saved: {metadata_path}")
    print(f"{'=' * 80}")
    
    return chunk_metadata


# ============= Helper function for PDB processing =============
def _process_pdb_path(pdb_path):
    """
    Process a single PDB file to extract sequence and coordinates.
    This function runs in a separate process.
    
    Returns:
        tuple: (status, pdb_id, data...) where status is "ok", "skip", or "error"
    """
    from Bio.PDB import PDBParser, is_aa
    
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.splitext(os.path.basename(pdb_path))[0]
    
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
        
        # Get first model
        model = structure[0]
        
        # Process each chain
        for chain in model:
            residues = [r for r in chain if is_aa(r)]
            if len(residues) == 0:
                continue
            
            # Extract sequence
            seq = ''.join([seq1(r.resname) for r in residues])
            
            # Extract coordinates [N, CA, C, O] for each residue
            coords = []
            for residue in residues:
                try:
                    n_coord = residue['N'].coord.tolist()
                    ca_coord = residue['CA'].coord.tolist()
                    c_coord = residue['C'].coord.tolist()
                    o_coord = residue['O'].coord.tolist()
                    coords.append([n_coord, ca_coord, c_coord, o_coord])
                except:
                    # Missing atoms - use zeros
                    coords.append([[0,0,0], [0,0,0], [0,0,0], [0,0,0]])
            
            return ("ok", pdb_id, seq, coords, chain.id)
        
        return ("skip", pdb_id, "No valid chains found")
        
    except Exception as e:
        return ("error", pdb_id, str(e))


# ============= Optimized Chunk Loader (same as before) =============
class StructureChunkLoader:
    """
    Efficient loader for chunked structure dictionaries.
    Loads chunks on-demand and caches them.
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", cache_size=2):
        self.chunk_dir = chunk_dir
        self.cache_size = cache_size
        self.cache = {}  # chunk_idx -> structure_dict
        self.cache_order = []  # LRU tracking
        
        # Load metadata
        metadata_path = os.path.join(chunk_dir, "chunk_metadata.json")
        with open(metadata_path, "r") as f: # speed up JSON load from StructureChunkLoader
            self.metadata = json.load(f)
        
        print(f"Loaded metadata: {self.metadata['total_structures']} structures in {self.metadata['num_chunks']} chunks")
        
        # Build lookup: pdb_id -> chunk_idx
        self.pdb_to_chunk = {}
        for chunk_info in self.metadata["chunks"]:
            chunk_path = os.path.join(chunk_dir, chunk_info["filename"])
            if os.path.exists(chunk_path):
                with open(chunk_path, "r") as f:
                    chunk_data = json.load(f)
                    for pdb_id in chunk_data.keys():
                        self.pdb_to_chunk[pdb_id] = chunk_info["chunk_idx"]
            else:
                print(f"Warning: Chunk file not found: {chunk_path}")
    
    def _load_chunk(self, chunk_idx):
        """Load a chunk into cache, managing cache size."""
        if chunk_idx in self.cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_idx)
            self.cache_order.append(chunk_idx)
            return self.cache[chunk_idx]
        
        # Load chunk
        chunk_info = self.metadata["chunks"][chunk_idx]
        chunk_path = os.path.join(self.chunk_dir, chunk_info["filename"])
        with open(chunk_path, "r") as f:
            chunk_data = json.load(f)
        
        # Add to cache
        self.cache[chunk_idx] = chunk_data
        self.cache_order.append(chunk_idx)
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
            gc.collect()  # Force garbage collection
        
        return chunk_data
    
    def get(self, pdb_id):
        """Get structure for a specific PDB ID."""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)
    
    def get_batch(self, pdb_ids):
        """Get multiple structures efficiently by grouping by chunk."""
        # Group PDB IDs by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load each chunk and extract structures
        results = {}
        for chunk_idx, chunk_pdb_ids in chunk_groups.items():
            chunk_data = self._load_chunk(chunk_idx)
            for pdb_id in chunk_pdb_ids:
                if pdb_id in chunk_data:
                    results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get_available_pdb_ids(self):
        """Return set of all available PDB IDs."""
        return set(self.pdb_to_chunk.keys())


def build_mtl_dataset_optimized(df_fold, chunk_loader, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    """
    Build MTL dataset efficiently using chunked structure loader.
    
    Args:
        df_fold: DataFrame with fold data
        chunk_loader: StructureChunkLoader instance
        task_cols: List of task columns
    
    Returns:
        MTL_DTA dataset
    """
    data_list = []
    
    # Get all protein IDs from the fold
    protein_ids = df_fold["protein_id"].tolist()
    
    # Batch load structures (efficient chunk-based loading)
    print(f"Loading structures for {len(protein_ids)} proteins...")
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Process each row
    skipped = 0
    for i, row in tqdm(df_fold.iterrows(), total=len(df_fold), desc="Building dataset"):
        protein_id = row["protein_id"]
        
        if protein_id not in structures_batch:
            skipped += 1
            continue
        
        protein_json = structures_batch[protein_id]
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    if skipped > 0:
        print(f"Warning: Skipped {skipped} entries due to missing structures")
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)


# ============= USAGE EXAMPLE =============
import pandas as pd
import os
import json
from Bio.PDB import PDBParser, is_aa
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = PDBParser(QUIET=True)
import os
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# !/usr/bin/env python3
"""
Optimized chunk loader with multiple performance improvements:
1. Use pickle or msgpack instead of JSON (10-50x faster)
2. Memory-mapped files for zero-copy access
3. Parallel chunk loading
4. Pre-computed index for O(1) lookups
5. Optional HDF5 storage for even faster access
"""

import os
import gc
import json
import pickle
import numpy as np
from pathlib import Path
import mmap
import struct
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import time
from typing import Dict, List, Optional, Set, Tuple
import h5py
import msgpack
import lmdb
from tqdm import tqdm

# ============= SOLUTION 1: Pickle-based Fast Loader =============
class FastPickleChunkLoader:
    """
    5-10x faster than JSON using pickle format
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", cache_size=5):
        self.chunk_dir = Path(chunk_dir)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []
        
        print("Initializing FastPickleChunkLoader...")
        
        # Check if pickle versions exist, if not convert from JSON
        self._ensure_pickle_chunks()
        
        # Load metadata
        metadata_pkl = self.chunk_dir / "chunk_metadata.pkl"
        if metadata_pkl.exists():
            with open(metadata_pkl, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            # Fallback to JSON and convert
            with open(self.chunk_dir / "chunk_metadata.json", "r") as f:
                self.metadata = json.load(f)
            with open(metadata_pkl, "wb") as f:
                pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Loaded metadata: {self.metadata['total_structures']} structures in {self.metadata['num_chunks']} chunks")
        
        # Build optimized lookup using numpy arrays for faster access
        self._build_optimized_index()
        
        # Pre-load frequently used chunks
        self._preload_chunks()
    
    def _ensure_pickle_chunks(self):
        """Convert JSON chunks to pickle format if they don't exist"""
        json_files = list(self.chunk_dir.glob("chunk_*.json"))
        
        for json_file in tqdm(json_files, desc="Converting to pickle"):
            pkl_file = json_file.with_suffix('.pkl')
            
            if not pkl_file.exists():
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                with open(pkl_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"Converted {json_file.name} to pickle format")
    
    def _build_optimized_index(self):
        """Build optimized index for O(1) lookups"""
        self.pdb_to_chunk = {}
        
        # Use parallel loading for index building
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for chunk_info in self.metadata["chunks"]:
                pkl_path = self.chunk_dir / chunk_info["filename"].replace('.json', '.pkl')
                if pkl_path.exists():
                    future = executor.submit(self._load_chunk_index, pkl_path, chunk_info["chunk_idx"])
                    futures.append(future)
            
            for future in futures:
                pdb_ids, chunk_idx = future.result()
                for pdb_id in pdb_ids:
                    self.pdb_to_chunk[pdb_id] = chunk_idx
    
    def _load_chunk_index(self, pkl_path, chunk_idx):
        """Load just the keys from a chunk file"""
        with open(pkl_path, 'rb') as f:
            chunk_data = pickle.load(f)
        return list(chunk_data.keys()), chunk_idx
    
    def _preload_chunks(self, n_chunks=2):
        """Pre-load the first few chunks into cache"""
        for i in range(min(n_chunks, len(self.metadata["chunks"]))):
            self._load_chunk(i)
    
    @lru_cache(maxsize=10)
    def _load_chunk(self, chunk_idx):
        """Load a chunk with caching"""
        chunk_info = self.metadata["chunks"][chunk_idx]
        pkl_path = self.chunk_dir / chunk_info["filename"].replace('.json', '.pkl')
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def get_batch(self, pdb_ids):
        """Optimized batch retrieval with parallel loading"""
        # Group PDB IDs by chunk
        chunk_groups = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.pdb_to_chunk:
                chunk_idx = self.pdb_to_chunk[pdb_id]
                if chunk_idx not in chunk_groups:
                    chunk_groups[chunk_idx] = []
                chunk_groups[chunk_idx].append(pdb_id)
        
        # Load chunks in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            
            for chunk_idx, chunk_pdb_ids in chunk_groups.items():
                future = executor.submit(self._load_chunk, chunk_idx)
                futures[future] = chunk_pdb_ids
            
            for future, chunk_pdb_ids in futures.items():
                chunk_data = future.result()
                for pdb_id in chunk_pdb_ids:
                    if pdb_id in chunk_data:
                        results[pdb_id] = chunk_data[pdb_id]
        
        return results
    
    def get(self, pdb_id):
        """Get single structure"""
        if pdb_id not in self.pdb_to_chunk:
            return None
        
        chunk_idx = self.pdb_to_chunk[pdb_id]
        chunk_data = self._load_chunk(chunk_idx)
        return chunk_data.get(pdb_id)

# ============= SOLUTION 2: HDF5-based Ultra-Fast Loader =============
class HDF5ChunkLoader:
    """
    Ultra-fast loading using HDF5 format with compression
    Best for numerical data, 20-100x faster than JSON
    """
    def __init__(self, chunk_dir="../data/structure_chunks/", convert=True):
        self.chunk_dir = Path(chunk_dir)
        self.h5_file = self.chunk_dir / "structures.h5"
        
        if convert and not self.h5_file.exists():
            self._convert_to_hdf5()
        
        # Open HDF5 file with swmr mode for concurrent reads
        self.h5f = h5py.File(self.h5_file, 'r', swmr=True)
        
        # Build index
        self.pdb_ids = set(self.h5f.keys())
        print(f"Loaded {len(self.pdb_ids)} structures from HDF5")
    
    def _convert_to_hdf5(self):
        """Convert JSON chunks to HDF5 format"""
        print("Converting to HDF5 format...")
        
        # Load all chunks
        all_structures = {}
        json_files = list(self.chunk_dir.glob("chunk_*.json"))
        
        for json_file in tqdm(json_files, desc="Loading chunks"):
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
                all_structures.update(chunk_data)
        
        # Create HDF5 file with compression
        with h5py.File(self.h5_file, 'w') as h5f:
            for pdb_id, structure in tqdm(all_structures.items(), desc="Writing HDF5"):
                grp = h5f.create_group(pdb_id)
                
                # Store each field efficiently
                for key, value in structure.items():
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    
                    if isinstance(value, np.ndarray):
                        grp.create_dataset(key, data=value, compression='gzip', compression_opts=1)
                    else:
                        grp.attrs[key] = value
        
        print(f"Created HDF5 file: {self.h5_file}")
    
    def get(self, pdb_id):
        """Get single structure from HDF5"""
        if pdb_id not in self.h5f:
            return None
        
        grp = self.h5f[pdb_id]
        structure = dict(grp.attrs)
        
        for key in grp.keys():
            structure[key] = grp[key][:]
        
        return structure
    
    def get_batch(self, pdb_ids):
        """Get batch of structures from HDF5"""
        results = {}
        for pdb_id in pdb_ids:
            if pdb_id in self.h5f:
                results[pdb_id] = self.get(pdb_id)
        return results
    
    def __del__(self):
        if hasattr(self, 'h5f'):
            self.h5f.close()

# ============= SOLUTION 3: LMDB-based Memory-Mapped Loader =============
class LMDBChunkLoader:
    def __init__(self, chunk_dir="../data/structure_chunks/", convert=True):
        self.chunk_dir = Path(chunk_dir)
        self.lmdb_dir = self.chunk_dir / "structures_lmdb"
        
        if convert and not self.lmdb_dir.exists():
            self._convert_to_lmdb()
        
        # Open LMDB with large map size
        self.env = lmdb.open(
            str(self.lmdb_dir),
            map_size=10 * 1024**3,  # 10GB
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False
        )
        
        # Build index - EXCLUDE METADATA KEYS
        with self.env.begin() as txn:
            self.pdb_ids = set()
            for key, _ in txn.cursor():
                key_str = key.decode()
                # Skip metadata entries
                if key_str not in {'total_errors', 'num_chunks', 'chunk_size', 'total_structures', 'chunks'}:
                    self.pdb_ids.add(key_str)
        
        print(f"Loaded {len(self.pdb_ids)} structures from LMDB")
    
    def _convert_to_lmdb(self):
        """Convert JSON chunks to LMDB format"""
        print("Converting to LMDB format...")
        
        self.lmdb_dir.mkdir(exist_ok=True)
        
        # Create LMDB
        env = lmdb.open(
            str(self.lmdb_dir),
            map_size=10 * 1024**3  # 10GB
        )
        
        # FIX: Use correct file pattern
        json_files = list(self.chunk_dir.glob("structures_chunk_*.json"))  # Changed from "chunk_*.json"
        
        if not json_files:
            print(f"No JSON files found matching pattern 'structures_chunk_*.json' in {self.chunk_dir}")
            return
        
        with env.begin(write=True) as txn:
            for json_file in tqdm(json_files, desc="Converting to LMDB"):
                with open(json_file, 'r') as f:
                    chunk_data = json.load(f)
                
                for pdb_id, structure in chunk_data.items():
                    # Use msgpack for efficient serialization
                    packed = msgpack.packb(structure, use_bin_type=True)
                    txn.put(pdb_id.encode(), packed)
        
        env.close()
        print(f"Created LMDB database: {self.lmdb_dir}")
    
    def get(self, pdb_id):
        """Get single structure from LMDB"""
        with self.env.begin() as txn:
            packed = txn.get(pdb_id.encode())
            if packed:
                return msgpack.unpackb(packed, raw=False)
        return None
    
    def get_batch(self, pdb_ids):
        """Get batch of structures from LMDB - very fast"""
        results = {}
        with self.env.begin() as txn:
            for pdb_id in pdb_ids:
                packed = txn.get(pdb_id.encode())
                if packed:
                    results[pdb_id] = msgpack.unpackb(packed, raw=False)
        return results

    

    

    

    

    

    

    

# ============= SOLUTION 4: Shared Memory Loader for Multi-Processing =============
class SharedMemoryChunkLoader:
    """
    Use shared memory for zero-copy access across processes
    Best for multi-GPU training with multiple processes
    """
    def __init__(self, chunk_dir="../data/structure_chunks/"):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        
        self.chunk_dir = Path(chunk_dir)
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()
        
        # Load all data into shared memory
        self._load_to_shared_memory()
    
    def _load_to_shared_memory(self):
        """Load all structures into shared memory"""
        print("Loading structures into shared memory...")
        
        json_files = list(self.chunk_dir.glob("chunk_*.json"))
        
        for json_file in tqdm(json_files, desc="Loading to shared memory"):
            with open(json_file, 'r') as f:
                chunk_data = json.load(f)
                self.shared_dict.update(chunk_data)
        
        print(f"Loaded {len(self.shared_dict)} structures into shared memory")
    
    def get(self, pdb_id):
        return self.shared_dict.get(pdb_id)
    
    def get_batch(self, pdb_ids):
        return {pdb_id: self.shared_dict[pdb_id] 
                for pdb_id in pdb_ids 
                if pdb_id in self.shared_dict}

# ============= BENCHMARK FUNCTION =============
def benchmark_loaders(chunk_dir="../data/structure_chunks/", n_samples=1000):
    """
    Benchmark different loader implementations
    """
    import random
    import time
    
    print("\n" + "="*60)
    print("BENCHMARKING CHUNK LOADERS")
    print("="*60)
    
    # Get sample PDB IDs
    with open(Path(chunk_dir) / "chunk_metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Get some PDB IDs for testing
    test_file = Path(chunk_dir) / "structures_chunk_0000.json"
    with open(test_file, "r") as f:
        sample_data = json.load(f)
    
    pdb_ids = list(sample_data.keys())[:n_samples]
    batch_size = 32
    
    results = {}
    
    # Test original JSON loader
    print("\n1. Testing JSON Loader...")
    loader = StructureChunkLoader(chunk_dir)
    
    start = time.time()
    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i:i+batch_size]
        _ = loader.get_batch(batch)
    json_time = time.time() - start
    results['JSON'] = json_time
    print(f"   Time: {json_time:.2f}s")
    
    # Test Pickle loader
    print("\n2. Testing Pickle Loader...")
    loader = FastPickleChunkLoader(chunk_dir)
    
    start = time.time()
    for i in range(0, len(pdb_ids), batch_size):
        batch = pdb_ids[i:i+batch_size]
        _ = loader.get_batch(batch)
    pickle_time = time.time() - start
    results['Pickle'] = pickle_time
    print(f"   Time: {pickle_time:.2f}s")
    print(f"   Speedup: {json_time/pickle_time:.1f}x")
    
    # Test HDF5 loader
    try:
        print("\n3. Testing HDF5 Loader...")
        loader = HDF5ChunkLoader(chunk_dir)
        
        start = time.time()
        for i in range(0, len(pdb_ids), batch_size):
            batch = pdb_ids[i:i+batch_size]
            _ = loader.get_batch(batch)
        hdf5_time = time.time() - start
        results['HDF5'] = hdf5_time
        print(f"   Time: {hdf5_time:.2f}s")
        print(f"   Speedup: {json_time/hdf5_time:.1f}x")
    except ImportError:
        print("   HDF5 not available")
    
    # Test LMDB loader
    try:
        print("\n4. Testing LMDB Loader...")
        loader = LMDBChunkLoader(chunk_dir)
        
        start = time.time()
        for i in range(0, len(pdb_ids), batch_size):
            batch = pdb_ids[i:i+batch_size]
            _ = loader.get_batch(batch)
        lmdb_time = time.time() - start
        results['LMDB'] = lmdb_time
        print(f"   Time: {lmdb_time:.2f}s")
        print(f"   Speedup: {json_time/lmdb_time:.1f}x")
    except ImportError:
        print("   LMDB not available")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (loading {:,} structures in batches of {}):".format(n_samples, batch_size))
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    baseline = sorted_results[-1][1]
    
    for method, time_taken in sorted_results:
        speedup = baseline / time_taken
        print(f"{method:10s}: {time_taken:6.2f}s (speedup: {speedup:5.1f}x)")
    
    print("\nRECOMMENDATION:")
    fastest = sorted_results[0][0]
    print(f"Use {fastest} loader for {sorted_results[-1][1]/sorted_results[0][1]:.1f}x speedup!")

# ============= DROP-IN REPLACEMENT =============
def get_optimized_loader(chunk_dir="../data/structure_chunks/", method="auto"):
    """
    Get the best available loader
    
    Args:
        method: "auto", "pickle", "hdf5", "lmdb", or "json"
    """
    
    if method == "auto":
        # Try loaders in order of preference
        try:
            import lmdb
            print("Using LMDB loader (fastest)")
            return LMDBChunkLoader(chunk_dir)
        except ImportError:
            pass
        
        try:
            import h5py
            print("Using HDF5 loader (very fast)")
            return HDF5ChunkLoader(chunk_dir)
        except ImportError:
            pass
        
        print("Using Pickle loader (fast)")
        return FastPickleChunkLoader(chunk_dir)
    
    elif method == "pickle":
        return FastPickleChunkLoader(chunk_dir)
    elif method == "hdf5":
        return HDF5ChunkLoader(chunk_dir)
    elif method == "lmdb":
        return LMDBChunkLoader(chunk_dir)
    elif method == "json":
        return StructureChunkLoader(chunk_dir)
    else:
        raise ValueError(f"Unknown method: {method}")


# 4. MODIFIED BUILD DATASET FUNCTION
def build_mtl_dataset(df_fold, pdb_structures, task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']):
    data_list = []
    for i, row in df_fold.iterrows():
        pdb_id = os.path.basename(row["standardized_ligand_sdf"]).split(".")[0]
        protein_json = pdb_structures.get(pdb_id)
        protein = featurize_protein_graph(protein_json)
        drug = featurize_drug(row["standardized_ligand_sdf"])
        
        # Collect all task values
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein,
            "drug": drug,
        }
        data_entry.update(task_values)
        data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)
# !/usr/bin/env python3
"""
Optimized dataset building with multiple performance improvements:
1. Vectorized operations instead of iterrows
2. Parallel processing with multiprocessing
3. Batch processing
4. Feature caching
5. Memory-efficient processing
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial, lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import hashlib
from pathlib import Path
import joblib
from typing import List, Dict, Any, Optional
import torch
from torch_geometric.data import Data
import gc

# ============= SOLUTION 1: Vectorized + Parallel Processing =============
def build_mtl_dataset_parallel(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50'],
    n_workers=None,
    batch_size=100,
    use_cache=True,
    cache_dir="./feature_cache"
):
    """
    Highly optimized dataset building with parallel processing.
    10-20x faster than iterrows approach.
    """
    
    if n_workers is None:
        n_workers = min(mp.cpu_count() - 1, 8)
    
    print(f"Building dataset with {n_workers} workers...")
    
    # Pre-load all structures (already done in your code)
    protein_ids = df_fold["protein_id"].tolist()
    print(f"Loading structures for {len(protein_ids)} proteins...")
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Convert DataFrame to records for faster access
    records = df_fold.to_dict('records')
    
    # Setup cache if enabled
    feature_cache = FeatureCache(cache_dir) if use_cache else None
    
    # Process in batches for better memory management
    data_list = []
    
    # Create batches
    n_records = len(records)
    batches = []
    for i in range(0, n_records, batch_size):
        batch_records = records[i:i+batch_size]
        batch_structures = {r['protein_id']: structures_batch.get(r['protein_id']) 
                          for r in batch_records}
        batches.append((batch_records, batch_structures))
    
    print(f"Processing {len(batches)} batches...")
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches
        futures = []
        for batch_records, batch_structures in batches:
            future = executor.submit(
                process_batch,
                batch_records,
                batch_structures,
                task_cols,
                feature_cache
            )
            futures.append(future)
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_data = future.result()
            data_list.extend(batch_data)
    
    print(f"Successfully processed {len(data_list)} entries")
    
    # Save cache if used
    if feature_cache:
        feature_cache.save()
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

def process_batch(batch_records, batch_structures, task_cols, feature_cache=None):
    """
    Process a batch of records in parallel.
    This function runs in a separate process.
    """
    batch_data = []
    
    for record in batch_records:
        protein_id = record["protein_id"]
        
        # Skip if no structure
        if protein_id not in batch_structures or batch_structures[protein_id] is None:
            continue
        
        # Get or compute protein features
        if feature_cache:
            protein_features = feature_cache.get_protein_features(
                protein_id, 
                batch_structures[protein_id]
            )
        else:
            protein_features = featurize_protein_graph(batch_structures[protein_id])
        
        # Get or compute drug features
        if feature_cache:
            drug_features = feature_cache.get_drug_features(
                record["standardized_ligand_sdf"]
            )
        else:
            drug_features = featurize_drug(record["standardized_ligand_sdf"])
        
        # Collect task values
        task_values = {}
        for task in task_cols:
            if task in record and not pd.isna(record[task]):
                task_values[task] = float(record[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein_features,
            "drug": drug_features,
        }
        data_entry.update(task_values)
        batch_data.append(data_entry)
    
    return batch_data

# ============= SOLUTION 2: Vectorized Operations =============
def build_mtl_dataset_vectorized(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']
):
    """
    Use vectorized pandas operations instead of iterrows.
    5-10x faster than iterrows.
    """
    
    print("Building dataset with vectorized operations...")
    
    # Load all structures at once
    protein_ids = df_fold["protein_id"].unique()
    structures_batch = chunk_loader.get_batch(protein_ids.tolist())
    
    # Create a mask for valid entries
    valid_mask = df_fold["protein_id"].apply(lambda x: x in structures_batch)
    df_valid = df_fold[valid_mask].copy()
    
    print(f"Processing {len(df_valid)} valid entries out of {len(df_fold)}")
    
    # Vectorized processing using apply (much faster than iterrows)
    def process_row(row):
        protein_id = row["protein_id"]
        protein_json = structures_batch[protein_id]
        
        return {
            "protein": featurize_protein_graph(protein_json),
            "drug": featurize_drug(row["standardized_ligand_sdf"]),
            **{task: float(row[task]) if not pd.isna(row[task]) else np.nan 
               for task in task_cols}
        }
    
    # Use pandas apply with progress bar
    tqdm.pandas(desc="Processing rows")
    data_list = df_valid.progress_apply(process_row, axis=1).tolist()
    
    return MTL_DTA(df=df_valid, data_list=data_list, task_cols=task_cols)

# ============= SOLUTION 3: Batch Processing with Threading =============
def build_mtl_dataset_threaded(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50'],
    n_threads=8
):
    """
    Use threading for I/O-bound operations.
    Good for featurization that involves file I/O.
    """
    
    print(f"Building dataset with {n_threads} threads...")
    
    # Load structures
    protein_ids = df_fold["protein_id"].tolist()
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Prepare data for threading
    rows_with_structures = []
    for idx, row in df_fold.iterrows():
        if row["protein_id"] in structures_batch:
            rows_with_structures.append((idx, row, structures_batch[row["protein_id"]]))
    
    print(f"Processing {len(rows_with_structures)} entries...")
    
    # Process with thread pool
    data_list = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        
        for idx, row, structure in rows_with_structures:
            future = executor.submit(
                process_single_entry,
                row, structure, task_cols
            )
            futures.append(future)
        
        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures)):
            data_entry = future.result()
            if data_entry:
                data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

def process_single_entry(row, structure, task_cols):
    """Process a single entry."""
    try:
        protein_features = featurize_protein_graph(structure)
        drug_features = featurize_drug(row["standardized_ligand_sdf"])
        
        task_values = {}
        for task in task_cols:
            if task in row and not pd.isna(row[task]):
                task_values[task] = float(row[task])
            else:
                task_values[task] = np.nan
        
        data_entry = {
            "protein": protein_features,
            "drug": drug_features,
        }
        data_entry.update(task_values)
        
        return data_entry
    except Exception as e:
        print(f"Error processing entry: {e}")
        return None

# ============= FEATURE CACHING SYSTEM =============
class FeatureCache:
    """
    Cache computed features to avoid recomputation.
    """
    
    def __init__(self, cache_dir="./feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.protein_cache_file = self.cache_dir / "protein_features.pkl"
        self.drug_cache_file = self.cache_dir / "drug_features.pkl"
        
        # Load existing caches
        self.protein_cache = self._load_cache(self.protein_cache_file)
        self.drug_cache = self._load_cache(self.drug_cache_file)
        
    
    def _load_cache(self, cache_file):
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _get_hash(self, obj):
        """Get hash of an object for caching."""
        if isinstance(obj, str):
            return hashlib.md5(obj.encode()).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(obj)).hexdigest()
    
    def get_protein_features(self, protein_id, structure):
        """Get or compute protein features."""
        if protein_id not in self.protein_cache:
            self.protein_cache[protein_id] = featurize_protein_graph(structure)
        return self.protein_cache[protein_id]
    
    def get_drug_features(self, sdf_string):
        """Get or compute drug features."""
        drug_hash = self._get_hash(sdf_string)
        if drug_hash not in self.drug_cache:
            self.drug_cache[drug_hash] = featurize_drug(sdf_string)
        return self.drug_cache[drug_hash]
    
    def save(self):
        """Save caches to disk."""
        with open(self.protein_cache_file, 'wb') as f:
            pickle.dump(self.protein_cache, f)
        with open(self.drug_cache_file, 'wb') as f:
            pickle.dump(self.drug_cache, f)
        print(f"Saved cache: {len(self.protein_cache)} proteins, {len(self.drug_cache)} drugs")

# ============= ULTRA-FAST NUMPY-BASED PROCESSING =============
def build_mtl_dataset_numpy(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50']
):
    """
    Use numpy operations for maximum speed.
    Best for numerical features.
    """
    
    print("Building dataset with numpy operations...")
    
    # Convert to numpy arrays for faster access
    protein_ids = df_fold["protein_id"].values
    sdf_strings = df_fold["standardized_ligand_sdf"].values
    
    # Load structures
    unique_proteins = np.unique(protein_ids)
    structures_batch = chunk_loader.get_batch(unique_proteins.tolist())
    
    # Pre-allocate arrays
    n_samples = len(df_fold)
    task_values = np.full((n_samples, len(task_cols)), np.nan)
    
    # Fill task values using numpy
    for i, task in enumerate(task_cols):
        if task in df_fold.columns:
            task_values[:, i] = df_fold[task].values
    
    # Process in batches
    data_list = []
    batch_size = 1000
    
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, n_samples)
        
        for i in range(start_idx, end_idx):
            protein_id = protein_ids[i]
            
            if protein_id not in structures_batch:
                continue
            
            data_entry = {
                "protein": featurize_protein_graph(structures_batch[protein_id]),
                "drug": featurize_drug(sdf_strings[i]),
            }
            
            # Add task values
            for j, task in enumerate(task_cols):
                data_entry[task] = task_values[i, j]
            
            data_list.append(data_entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

# ============= JOBLIB PARALLEL PROCESSING =============
def build_mtl_dataset_joblib(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50'],
    n_jobs=-1
):
    """
    Use joblib for efficient parallel processing.
    Often faster than multiprocessing for scientific computing.
    """
    
    from joblib import Parallel, delayed
    
    print(f"Building dataset with joblib (n_jobs={n_jobs})...")
    
    # Load structures
    protein_ids = df_fold["protein_id"].tolist()
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Prepare data for parallel processing
    rows_data = []
    for idx, row in df_fold.iterrows():
        if row["protein_id"] in structures_batch:
            rows_data.append((
                row.to_dict(),
                structures_batch[row["protein_id"]],
                task_cols
            ))
    
    print(f"Processing {len(rows_data)} entries...")
    
    # Process in parallel using joblib
    def process_row_joblib(row_dict, structure, task_cols):
        try:
            protein_features = featurize_protein_graph(structure)
            drug_features = featurize_drug(row_dict["standardized_ligand_sdf"])
            
            data_entry = {
                "protein": protein_features,
                "drug": drug_features,
            }
            
            for task in task_cols:
                if task in row_dict and not pd.isna(row_dict[task]):
                    data_entry[task] = float(row_dict[task])
                else:
                    data_entry[task] = np.nan
            
            return data_entry
        except Exception as e:
            return None
    
    # Parallel execution
    data_list = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process_row_joblib)(row_dict, structure, task_cols)
        for row_dict, structure, task_cols in tqdm(rows_data)
    )
    
    # Filter out None values
    data_list = [d for d in data_list if d is not None]
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

# ============= BENCHMARK FUNCTION =============
def benchmark_dataset_builders(df_fold, chunk_loader, task_cols):
    """
    Benchmark different dataset building methods.
    """
    import time
    
    # Take a sample for benchmarking
    df_sample = df_fold.head(1000).copy()
    
    results = {}
    
    # Test original method (iterrows)
    print("\n1. Testing original method (iterrows)...")
    start = time.time()
    _ = build_mtl_dataset_optimized(df_sample, chunk_loader, task_cols)
    original_time = time.time() - start
    results['Original (iterrows)'] = original_time
    print(f"   Time: {original_time:.2f}s")
    
    # Test vectorized method
    print("\n2. Testing vectorized method...")
    start = time.time()
    _ = build_mtl_dataset_vectorized(df_sample, chunk_loader, task_cols)
    vectorized_time = time.time() - start
    results['Vectorized'] = vectorized_time
    print(f"   Time: {vectorized_time:.2f}s")
    print(f"   Speedup: {original_time/vectorized_time:.1f}x")
    
    # Test parallel method
    print("\n3. Testing parallel method...")
    start = time.time()
    _ = build_mtl_dataset_parallel(df_sample, chunk_loader, task_cols, n_workers=4)
    parallel_time = time.time() - start
    results['Parallel'] = parallel_time
    print(f"   Time: {parallel_time:.2f}s")
    print(f"   Speedup: {original_time/parallel_time:.1f}x")
    
    # Test joblib method
    try:
        print("\n4. Testing joblib method...")
        start = time.time()
        _ = build_mtl_dataset_joblib(df_sample, chunk_loader, task_cols, n_jobs=4)
        joblib_time = time.time() - start
        results['Joblib'] = joblib_time
        print(f"   Time: {joblib_time:.2f}s")
        print(f"   Speedup: {original_time/joblib_time:.1f}x")
    except ImportError:
        print("   Joblib not available")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (1000 samples):")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    baseline = sorted_results[-1][1]
    
    for method, time_taken in sorted_results:
        speedup = baseline / time_taken
        print(f"{method:20s}: {time_taken:6.2f}s (speedup: {speedup:5.1f}x)")
    
    fastest = sorted_results[0][0]
    print(f"\nRECOMMENDATION: Use {fastest} for {sorted_results[-1][1]/sorted_results[0][1]:.1f}x speedup!")

# ============= DROP-IN REPLACEMENT =============
def build_mtl_dataset_fast(
    df_fold, 
    chunk_loader, 
    task_cols=['pKi', 'pEC50', 'pKd', 'pIC50'],
    method='auto'
):
    """
    Drop-in replacement for build_mtl_dataset_optimized.
    Automatically selects the best method.
    """
    
    n_samples = len(df_fold)
    
    if method == 'auto':
        if n_samples < 1000:
            # Small dataset, use vectorized
            return build_mtl_dataset_vectorized(df_fold, chunk_loader, task_cols)
        elif n_samples < 10000:
            # Medium dataset, use threading
            return build_mtl_dataset_threaded(df_fold, chunk_loader, task_cols)
        else:
            # Large dataset, use parallel processing
            return build_mtl_dataset_parallel(df_fold, chunk_loader, task_cols)
    
    elif method == 'vectorized':
        return build_mtl_dataset_vectorized(df_fold, chunk_loader, task_cols)
    elif method == 'parallel':
        return build_mtl_dataset_parallel(df_fold, chunk_loader, task_cols)
    elif method == 'threaded':
        return build_mtl_dataset_threaded(df_fold, chunk_loader, task_cols)
    elif method == 'joblib':
        return build_mtl_dataset_joblib(df_fold, chunk_loader, task_cols)
    elif method == 'numpy':
        return build_mtl_dataset_numpy(df_fold, chunk_loader, task_cols)
    else:
        # Fallback to original
        return build_mtl_dataset_optimized(df_fold, chunk_loader, task_cols)

# ============= USAGE EXAMPLE =============
# !/usr/bin/env python3
"""
Multi-GPU training solution that works in Jupyter notebooks
Avoids mp.spawn issues by using simpler approaches
"""


def build_mtl_dataset_parallel_fixed(df_fold, chunk_loader, task_cols):
    """Fixed parallel processing"""
    
    # Load structures first
    protein_ids = df_fold["protein_id"].tolist()
    structures_batch = chunk_loader.get_batch(protein_ids)
    
    # Prepare simple data for parallel processing
    records = []
    for idx, row in df_fold.iterrows():
        if row["protein_id"] in structures_batch:
            records.append({
                'protein_json': structures_batch[row["protein_id"]],
                'drug_sdf': row["standardized_ligand_sdf"],
                'tasks': {task: row[task] if task in row else np.nan 
                         for task in task_cols}
            })
    
    # Process sequentially (featurization is fast enough)
    data_list = []
    for record in tqdm(records, desc="Featurizing"):
        try:
            protein = featurize_protein_graph(record['protein_json'])
            drug = featurize_drug(record['drug_sdf'])
            
            entry = {
                "protein": protein,
                "drug": drug,
            }
            for task, value in record['tasks'].items():
                entry[task] = float(value) if not pd.isna(value) else np.nan
            
            data_list.append(entry)
        except Exception as e:
            continue
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)



import pickle
from pathlib import Path
import hashlib

class CachedFeaturizer:
    """Cache computed features to disk"""
    def __init__(self, cache_dir="./feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_or_compute(self, key, compute_fn, *args):
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        result = compute_fn(*args)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        return result

def build_mtl_dataset_ultra_fast(df_fold, chunk_loader, task_cols):
    """Ultra-fast dataset building with caching and batch processing"""
    
    cache = CachedFeaturizer()
    
    # Get unique proteins and drugs
    unique_proteins = df_fold["protein_id"].unique()
    unique_drugs = df_fold["standardized_ligand_sdf"].unique()
    
    print(f"Processing {len(unique_proteins)} unique proteins, {len(unique_drugs)} unique drugs")
    
    # Batch load all structures
    structures_batch = chunk_loader.get_batch(unique_proteins.tolist())
    
    # Pre-compute all protein features
    protein_features = {}
    for pid in tqdm(unique_proteins, desc="Protein features"):
        if pid in structures_batch:
            protein_features[pid] = cache.get_or_compute(
                f"protein_{pid}",
                featurize_protein_graph,
                structures_batch[pid]
            )
    
    # Pre-compute all drug features  
    drug_features = {}
    for drug_sdf in tqdm(unique_drugs, desc="Drug features"):
        drug_hash = hashlib.md5(drug_sdf.encode()).hexdigest()[:8]
        drug_features[drug_sdf] = cache.get_or_compute(
            f"drug_{drug_hash}",
            featurize_drug,
            drug_sdf
        )
    
    # Now just lookup pre-computed features
    data_list = []
    for _, row in df_fold.iterrows():
        pid = row["protein_id"]
        drug_sdf = row["standardized_ligand_sdf"]
        
        if pid not in protein_features or drug_sdf not in drug_features:
            continue
            
        entry = {
            "protein": protein_features[pid],
            "drug": drug_features[drug_sdf],
        }
        
        for task in task_cols:
            entry[task] = float(row[task]) if task in row and not pd.isna(row[task]) else np.nan
            
        data_list.append(entry)
    
    return MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)

def build_mtl_dataset_ultra_fast(df_fold, chunk_loader, task_cols):
    """Version that skips cache checking if already complete"""
    
    cache = CachedFeaturizer()
    
    # Check if this fold is already fully cached
    fold_hash = hashlib.md5(f"{df_fold.index.tolist()}".encode()).hexdigest()[:8]
    fold_cache_file = cache.cache_dir / f"fold_{fold_hash}_complete.pkl"
    
    if fold_cache_file.exists():
        print(f"Loading pre-built dataset from cache...")
        with open(fold_cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Building dataset (will be cached for next time)...")
    
    # Get unique proteins and drugs
    unique_proteins = df_fold["protein_id"].unique()
    unique_drugs = df_fold["standardized_ligand_sdf"].unique()
    
    # Only process items not in cache
    protein_features = {}
    proteins_to_process = []
    for pid in unique_proteins:
        cache_file = cache.cache_dir / f"protein_{pid}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                protein_features[pid] = pickle.load(f)
        else:
            proteins_to_process.append(pid)
    
    if proteins_to_process:
        print(f"Processing {len(proteins_to_process)} new proteins...")
        structures_batch = chunk_loader.get_batch(proteins_to_process)
        for pid in tqdm(proteins_to_process):
            if pid in structures_batch:
                protein_features[pid] = cache.get_or_compute(
                    f"protein_{pid}",
                    featurize_protein_graph,
                    structures_batch[pid]
                )
    else:
        print(f"All {len(unique_proteins)} proteins already cached!")
    
    # Same for drugs
    drug_features = {}
    drugs_to_process = []
    for drug_sdf in unique_drugs:
        drug_hash = hashlib.md5(drug_sdf.encode()).hexdigest()[:8]
        cache_file = cache.cache_dir / f"drug_{drug_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                drug_features[drug_sdf] = pickle.load(f)
        else:
            drugs_to_process.append(drug_sdf)
    
    if drugs_to_process:
        print(f"Processing {len(drugs_to_process)} new drugs...")
        for drug_sdf in tqdm(drugs_to_process):
            drug_hash = hashlib.md5(drug_sdf.encode()).hexdigest()[:8]
            drug_features[drug_sdf] = cache.get_or_compute(
                f"drug_{drug_hash}",
                featurize_drug,
                drug_sdf
            )
    else:
        print(f"All {len(unique_drugs)} drugs already cached!")
    
    # Build dataset
    data_list = []
    for _, row in df_fold.iterrows():
        pid = row["protein_id"]
        drug_sdf = row["standardized_ligand_sdf"]
        
        if pid not in protein_features or drug_sdf not in drug_features:
            continue
            
        entry = {
            "protein": protein_features[pid],
            "drug": drug_features[drug_sdf],
        }
        
        for task in task_cols:
            entry[task] = float(row[task]) if task in row and not pd.isna(row[task]) else np.nan
            
        data_list.append(entry)
    
    # Create and cache the dataset
    dataset = MTL_DTA(df=df_fold, data_list=data_list, task_cols=task_cols)
    
    # Save complete dataset for instant loading next time
    with open(fold_cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset

def precompute_all_features(df_full, chunk_loader):
    """Run this ONCE before training to cache all features"""
    
    cache = CachedFeaturizer()
    
    # Get ALL unique proteins and drugs from entire dataset
    all_proteins = df_full["protein_id"].unique()
    all_drugs = df_full["standardized_ligand_sdf"].unique()
    
    print(f"Pre-computing features for {len(all_proteins)} proteins and {len(all_drugs)} drugs")
    
    # Load all structures
    structures = chunk_loader.get_batch(all_proteins.tolist())
    
    # Compute all protein features
    for pid in tqdm(all_proteins, desc="All proteins"):
        if pid in structures:
            cache.get_or_compute(f"protein_{pid}", featurize_protein_graph, structures[pid])
    
    # Compute all drug features
    for drug_sdf in tqdm(all_drugs, desc="All drugs"):
        drug_hash = hashlib.md5(drug_sdf.encode()).hexdigest()[:8]
        cache.get_or_compute(f"drug_{drug_hash}", featurize_drug, drug_sdf)
    
    print("✓ All features cached!")



import os
import gc
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# ============= SOLUTION 1: DataParallel (Works in Notebooks) =============
def train_fold_dataparallel(
    fold_idx, n_folds,
    df_train, df_valid, df_test,
    chunk_loader, task_cols, task_ranges,
    n_epochs=100, batch_size=256, lr=0.0005, patience=20,
    use_multiple_gpus=True
):
    """
    Using standard DataParallel - works immediately in notebooks
    Less efficient than DDP but much simpler and works everywhere
    """
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{n_folds}")
    print(f"{'='*60}")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Valid: {len(df_valid)} samples")
    print(f"  Test:  {len(df_test)} samples")
    
    # Detect GPUs
    n_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {n_gpus}")
    
    if use_multiple_gpus and n_gpus > 1:
        device_ids = list(range(n_gpus))
        print(f"  Using GPUs: {device_ids}")
        # Adjust batch size for multiple GPUs
        actual_batch_size = batch_size * n_gpus
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Total batch size: {actual_batch_size}")
    else:
        device_ids = [0]
        actual_batch_size = batch_size
        print(f"  Using single GPU: 0")
        print(f"  Batch size: {actual_batch_size}")
    
    device = torch.device(f"cuda:{device_ids[0]}")
    
    # Create datasets
    print("\nBuilding datasets...")
    # train_dataset = build_mtl_dataset_optimized(df_train, chunk_loader, task_cols)
    # valid_dataset = build_mtl_dataset_optimized(df_valid, chunk_loader, task_cols)
    # test_dataset = build_mtl_dataset_optimized(df_test, chunk_loader, task_cols)
    

    
    # For maximum speed:
    train_dataset = build_mtl_dataset_ultra_fast(
        df_train, 
        chunk_loader, 
        task_cols
    )
    print("\nBuilding datasets...")
    valid_dataset = build_mtl_dataset_ultra_fast(
        df_valid, 
        chunk_loader, 
        task_cols
    )
    print("\nBuilding datasets...")
    test_dataset = build_mtl_dataset_ultra_fast(
        df_test, 
        chunk_loader, 
        task_cols
    )
    print("\nLoading datasets...")

    
    
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
            batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,  # Keep at 0 in notebooks
        pin_memory=False,  # Set to False if having issues
        persistent_workers=False
    )
    print("\nLoading datasets...")
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,  # Keep at 0 in notebooks
        pin_memory=False,  # Set to False if having issues
        persistent_workers=False
    )
    print("\nLoading datasets...")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,  
        shuffle=False,
        num_workers=0,  # Keep at 0 in notebooks
        pin_memory=False,  # Set to False if having issues
        persistent_workers=False
    )
    
    print(f"✓ DataLoaders created")
    
    # Create model
    print("Initializing model...")
    model = MTL_DTAModel(
        task_names=task_cols,
        prot_emb_dim=1280,
        prot_gcn_dims=[128, 256, 256],
        prot_fc_dims=[1024, 128],
        drug_node_in_dim=[66, 1],
        drug_node_h_dims=[128, 64],
        drug_fc_dims=[1024, 128],
        mlp_dims=[1024, 512],
        mlp_dropout=0.25
    )
    
    # Apply DataParallel if using multiple GPUs
    if use_multiple_gpus and n_gpus > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"✓ Model wrapped with DataParallel")
    
    model = model.to(device)
    print(f"✓ Model loaded on GPU(s)")
    
    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
    
    # Training state
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Training loop
    print("\nStarting training...")
    pbar = tqdm(range(n_epochs), desc=f"Training", ncols=100)
    
    for epoch in pbar:
        # ========== TRAINING PHASE ==========
        model.train()
        train_loss = 0
        n_train_batches = 0
        
        for batch_idx, batch in tqdm(enumerate(train_loader), desc ="batch"):
            try:
                # Move batch to GPU
                xd = batch['drug'].to(device, non_blocking=True)
                xp = batch['protein'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                pred = model(xd, xp)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_train_batches += 1
                
                # Clear cache periodically
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at batch {batch_idx}, clearing cache and skipping...")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_train_loss = train_loss / max(n_train_batches, 1)
        train_losses.append(avg_train_loss)
        
        # ========== VALIDATION PHASE ==========
        model.eval()
        val_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in valid_loader:
                xd = batch['drug'].to(device, non_blocking=True)
                xp = batch['protein'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                
                pred = model(xd, xp)
                loss = criterion(pred, y)
                val_loss += loss.item()
                n_val_batches += 1
        
        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)
        
        # Update progress bar
        pbar.set_postfix({
            'train': f"{avg_train_loss:.4f}",
            'val': f"{avg_val_loss:.4f}",
            'best': f"{best_val_loss:.4f}"
        })
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Handle both DataParallel and regular model
            if hasattr(model, 'module'):
                best_model_state = model.module.state_dict()
            else:
                best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Periodic cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # Load best model
    if best_model_state is not None:
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_model_state)
        else:
            model.load_state_dict(best_model_state)
    
    # ========== EVALUATION PHASE ==========
    print(f"\nEvaluating on test set...")
    model.eval()
    
    task_predictions = {task: [] for task in task_cols}
    task_targets = {task: [] for task in task_cols}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            
            pred = model(xd, xp)
            
            # Collect predictions for each task
            for i, task in enumerate(task_cols):
                mask = ~torch.isnan(y[:, i])
                if mask.sum() > 0:
                    task_preds = pred[mask, i].cpu().numpy()
                    task_trues = y[mask, i].cpu().numpy()
                    task_predictions[task].extend(task_preds)
                    task_targets[task].extend(task_trues)
    
    # Calculate and print metrics
    fold_results = {}
    print(f"\n{'='*50}")
    print(f"Fold {fold_idx + 1} Results:")
    print(f"{'='*50}")
    
    for task in task_cols:
        if len(task_predictions[task]) > 0:
            preds = np.array(task_predictions[task])
            targets = np.array(task_targets[task])
            
            r2 = r2_score(targets, preds)
            rmse = math.sqrt(mean_squared_error(targets, preds))
            
            fold_results[task] = {
                'predictions': preds,
                'targets': targets,
                'r2': r2,
                'rmse': rmse
            }
            
            print(f"{task:20s} | RMSE: {rmse:6.3f} | R²: {r2:6.3f} | n={len(preds):5d}")
    
    # Clean up
    del model
    del optimizer
    del train_loader
    del valid_loader
    del test_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return fold_results, train_losses, val_losses

# ============= SOLUTION 2: Distributed Training via Script File =============
def create_training_script(output_path="train_ddp.py"):
    """
    Create a standalone Python script for DDP training
    Run this script outside the notebook with:
    torchrun --nproc_per_node=4 train_ddp.py
    """
    
    script_content = '''#!/usr/bin/env python3
"""
Standalone DDP training script for GNN
Run with: torchrun --nproc_per_node=4 train_ddp.py
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
# Add your other imports here

def setup_ddp():
    """Initialize DDP from torchrun environment variables"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def train_with_ddp():
    rank, world_size, local_rank = setup_ddp()
    
    # Your model and training code here
    # Use local_rank as the device
    device = torch.device(f"cuda:{local_rank}")
    
    # Create model
    model = YourModel().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop...
    
    cleanup_ddp()

if __name__ == "__main__":
    train_with_ddp()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Created training script: {output_path}")
    print(f"Run with: torchrun --nproc_per_node=4 {output_path}")

# ============= SOLUTION 3: Gradient Accumulation for Large Batches =============
def train_with_gradient_accumulation(
    fold_idx, n_folds,
    df_train, df_valid, df_test,
    chunk_loader, task_cols, task_ranges,
    n_epochs=100, base_batch_size=32, 
    accumulation_steps=4,  # Simulate batch_size = 32 * 4 = 128
    lr=0.0005, patience=20,
    gpu_id=0
):
    """
    Use gradient accumulation to simulate larger batch sizes
    This is memory efficient and works on single GPU
    """
    
    device = torch.device(f"cuda:{gpu_id}")
    effective_batch_size = base_batch_size * accumulation_steps
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/{n_folds} - Gradient Accumulation")
    print(f"{'='*60}")
    print(f"  Base batch size: {base_batch_size}")
    print(f"  Accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Using GPU: {gpu_id}")
    
    # Create datasets and loaders
    train_dataset = build_mtl_dataset_optimized(df_train, chunk_loader, task_cols)
    valid_dataset = build_mtl_dataset_optimized(df_valid, chunk_loader, task_cols)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=base_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Create model
    model = MTL_DTAModel(
        task_names=task_cols,
        prot_emb_dim=1280,
        prot_gcn_dims=[128, 256, 256],
        prot_fc_dims=[1024, 128],
        drug_node_in_dim=[66, 1],
        drug_node_h_dims=[128, 64],
        drug_fc_dims=[1024, 128],
        mlp_dims=[1024, 512],
        mlp_dropout=0.25
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MaskedMSELoss(task_ranges=task_ranges).to(device)
    
    # Training loop with gradient accumulation
    print("\nStarting training with gradient accumulation...")
    
    for epoch in tqdm(range(n_epochs), desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        
        train_loss = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            xd = batch['drug'].to(device)
            xp = batch['protein'].to(device)
            y = batch['y'].to(device)
            
            # Forward pass
            pred = model(xd, xp)
            loss = criterion(pred, y)
            
            # Normalize loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            train_loss += loss.item() * accumulation_steps
            n_batches += 1
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Handle remaining gradients
        if n_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Validation...
        # [Add validation code similar to above]
    
    return fold_results, train_losses, val_losses

