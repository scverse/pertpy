import torch
import anndata
from tqdm import tqdm
from anndata import AnnData
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, WeightedRandomSampler
from pertpy.tools._perturbation_space._utils import MLP, PerturbationDataset, PerturbationClassifier

class DiscriminatorClassifierSpace:
    """Leveraging discriminator classifier: The idea here is that we fit either a regressor model for gene expression (see Supplemental Materials.
    here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19)
    and we use either coefficient of the model for each perturbation as a feature or train a classifier example
    (simple MLP or logistic regression and take the penultimate layer as feature space and apply pseudo bulking approach above)
    """

    def __call__(self, adata: AnnData, target_col: str = "perturbations", layer_key: str = None, hidden_dim: list = [512], dropout=0.0, batch_norm=True, *args, **kwargs):
        # TODO implement
        
        if layer_key is not None and layer_key not in adata.obs.columns:
            raise ValueError(
                f"Layer key {layer_key} not found in adata. {layer_key}"
            )
            
        # Handling classes
        n_classes = len(adata.obs[target_col].unique()) 
        labels = adata.obs[target_col]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        adata.obs['encoded_perturbations'] = encoded_labels
        
        # Split the data in train, test and validation
        X = list(range(0, adata.n_obs))
        y = adata.obs[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train) # 0.8 x 0.25 = 0.2
        
        train_dataset = PerturbationDataset(adata=adata[X_train], target_col='encoded_perturbations', label_col=target_col, layer_key=layer_key)
        val_dataset = PerturbationDataset(adata=adata[X_val], target_col='encoded_perturbations', label_col=target_col, layer_key=layer_key)
        test_dataset = PerturbationDataset(adata=adata[X_test], target_col='encoded_perturbations', label_col=target_col, layer_key=layer_key) # we don't need to pass y_test since the label selection is done inside
        
        # Fix class unbalance (always happens in pert. datasets)
        class_weights = 1.0 / torch.bincount(torch.tensor(train_dataset.labels.values))
        train_weights = class_weights[train_dataset.labels]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=128, sampler=train_sampler, num_workers=4)
        self.test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
        self.valid_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

        # Define the network
        sizes = [adata.n_vars] + hidden_dim + [n_classes]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = MLP(sizes=sizes, dropout=dropout, batch_norm=batch_norm, device=self.device)
        
        # Define total dataset and dataloader for getting embeddings
        total_dataset = PerturbationDataset(adata=adata, target_col='encoded_perturbations', label_col=target_col, layer_key=layer_key)
        self.total_dataloader = DataLoader(total_dataset, batch_size=512, shuffle=False, num_workers=4)

        return self
        
    def train(self, max_epochs: int =40, val_check: int = 5, patience: int = 2):
        
        self.trainer = pl.Trainer(
                    min_epochs=1,
                    max_epochs=max_epochs,
                    check_val_every_n_epoch=val_check, 
                    callbacks=[EarlyStopping(monitor="validation_loss", mode="min", patience=patience)], 
                    accelerator='cpu')
        
        self.model = PerturbationClassifier(model=self.net)
        
        self.trainer.fit(model=self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        self.trainer.test(model=self.model, dataloaders=self.test_dataloader)
        
    def get_embeddings(self):
        
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(self.total_dataloader):
                emb, y = self.model.get_embeddings(batch)
                batch_adata = AnnData(X=emb.cpu().numpy())
                batch_adata.obs['perturbations'] = y
                if i == 0:
                    pert_adata = batch_adata
                else:
                    pert_adata = anndata.concat([pert_adata, batch_adata])
                
        return pert_adata