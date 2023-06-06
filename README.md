# Improved Cylinder3D with PolarMix

## Description

Ce repository contient une amélioration de l'algorithme Cylinder3D utilisant la méthode PolarMix. Il s'agit d'une méthode d'augmentation de données se basant sur deux niveaux d'augmentation : 
- Scene-level swapping
- Instance-level rotate-pasting

Il s'agit d'un travail d'adaptation de l'article [PolarMix: A General Data Augmentation Technique for LiDAR Point Clouds](https://arxiv.org/abs/2208.00223) sur le repository [Cylinder3D](https://github.com/xinge008/Cylinder3D) (avec les mises à jour adaptés [pour CUDA](https://github.com/L-Reichardt/Cylinder3D-updated-CUDA)).

## Installation

Pour cloner le repository, utilisez la commande suivante :
```
git clone https://github.com/tinylinux/iasd_npm_polarmix.git
```

## Installation

### Requis

Pour installer ce repository, il faut disposer des versions suivantes avec une version de CUDA (à 11.3) :

- Python 3.8
- PyTorch == 1.8.0 --extra-index-url https://download.pytorch.org/whl/cu113
- yaml == 6.0
- strictyaml == 1.6.1
- Cython == 0.29.30
- tqdm == 4.64.0
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter) == cu113
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)
- [spconv-cu117 == 2.2.3](https://github.com/traveller59/spconv) (different CUDA versions available)
- numba == 0.55.2 (install last, as this will likely downgrade your numpy version automatically)

Vous pouvez installer cela à l'aide de ces commandes (pour CUDA à 11.3 et fonctionne aussi pour 10.2) 
```bash
pip install spconv-cu113
pip install torch-geometric torch-sparse torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu113.html
pip install strictyaml
```

## Utilisation

### Préparation du Dataset : SemanticKITTI

Avant d'exécuter le code, vous devez télécharger le dataset à partir du lien suivant : [dataset](https://perso.crans.org/rlali/cylinder_data.tar.gz)

Décompressez le fichier zip et placez-le dans le dossier `data/` à la racine du repository.

```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### Activer PolarMix

Pour activer la fonctionnalité PolarMix, vous devez ouvrir le fichier `dataloader/dataset_semantickitti_polarmix.py` et activer le flag correspondant en décommentant la ligne suivante :

```python
use_polarmix = True
use_alternative = False
```

Si vous souhaitez utiliser la fonctionnalité basique implémentée par Cylinder3D, il faudra activer le flag `use_alternative` comme suivant :

```python
use_polarmix = False
use_alternative = True
```

Il faut créer les dossiers `model_load_dir` et `model_save_dir` pour pouvoir sauvegarder les modèles générés.

```bash
mkdir model_load_dir
mkdir model_save_dir
```

Pour lancer l'algorithme, utilisez la commande suivante :

```bash
python train_cylinder_asym.py -y semantickitti_polarmix.yaml
```

Le fichier de configuration `semantickitti_polarmix.yaml` contient les paramètres d'entraînement ainsi que les paramètres de Dataset.

Il est recommandé d'utiliser les paramètres suivants :
- 40 epochs
- 0.00707 base LR with sqrt_k scaling rule (equals to original 0.001 at batchsize = 2, equals 0.00489 at batchsize = 24)
- AdamW with Weight Decay 0.001
- CosineDecay Schedule
- Batch Size 24 (Better result possible with lower batch size, batch size chosen for economical reasons.)