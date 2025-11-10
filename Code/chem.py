import xml.etree.ElementTree as ET
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

xml_path = "/DrugBank/full database.xml"
tree = ET.parse(xml_path)
root = tree.getroot()
ns = {"db": "http://www.drugbank.ca"}
#nohup python chem.py > chem.log 2>&1
records = []
for drug in root.findall("db:drug", ns):
    drugbank_id = drug.find("db:drugbank-id", ns).text
    name = drug.find("db:name", ns).text
    smiles = None
    for prop in drug.findall("db:calculated-properties/db:property", ns):
        kind = prop.find("db:kind", ns)
        if kind is not None and kind.text == "SMILES":
            smiles = prop.find("db:value", ns).text
    if smiles:
        records.append([drugbank_id, name, "DrugBank", smiles])

df = pd.DataFrame(records, columns=["drug_id", "drug_name", "source", "SMILES"])
df.to_csv("/drug_smiles.csv", index=False)
print("Extracted SMILES from DrugBank XML:", len(df))

# SMILES table
drug_smiles = pd.read_csv("/home/maoxiangyun/OldServer/知识图谱/DrugBank/drug_smiles.csv")

# upload ChemBERTa
tokenizer = AutoTokenizer.from_pretrained("PubChem10M_SMILES_BPE_396_250")
model = AutoModel.from_pretrained("PubChem10M_SMILES_BPE_396_250").to("cuda")

embeds = []
batch_size = 32
for i in tqdm(range(0, len(drug_smiles), batch_size)):
    batch = drug_smiles.iloc[i:i+batch_size]
    tokens = tokenizer(batch["SMILES"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    with torch.no_grad():
        outputs = model(**tokens)
        batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    embeds.append(batch_emb)

drug_smiles["structure_emb"] = np.vstack(embeds).tolist()
drug_smiles.to_parquet("/KG/drug_structure_emb.parquet", compression="snappy")
