# DeepNPA

DeepNPA is a deep learning model designed to predict **neuropeptide precursor cleavage sites**. It leverages the **ProtBert** pre-trained model combined with the **FocalLoss** loss function to effectively address the problem of data imbalance in neuropeptide prediction.

---

## Features

- Prediction of neuropeptide precursor cleavage sites.
- Utilizes **ProtBert** pre-trained embeddings for protein sequences.
- Implements **FocalLoss** to handle class imbalance.
- Provides Python-based implementation with easy integration into existing workflows.

---

## Requirements

- Python version: 3.11.9
- Dependencies:

```text
numpy==2.3.3
pandas==2.3.3
python-dateutil==2.9.0.post0
pytz==2025.2
tzdata==2025.2
openpyxl==3.1.5
six==1.17.0
torch==2.9.0
transformers==4.57.1
tokenizers==0.22.1
scikit-learn==1.7.2
scipy==1.16.2
joblib==1.5.2
threadpoolctl==3.6.0
requests==2.32.5
urllib3==2.5.0
certifi==2025.10.5
idna==3.11
charset-normalizer==3.4.4
PyYAML==6.0.3
tqdm==4.67.1
huggingface-hub==0.35.3
filelock==3.20.0
fsspec==2025.9.0
regex==2025.9.18
Jinja2==3.1.6
MarkupSafe==3.0.3
packaging==25.0
typing_extensions==4.15.0
beautifulsoup4==4.14.2
soupsieve==2.8
colorama==0.4.6
psutil==7.1.0
networkx==3.5
sympy==1.14.0
mpmath==1.3.0
ipython==9.6.0
pillow==11.3.0
tornado==6.5.2
