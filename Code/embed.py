import os
import pandas as pd
import numpy as np
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator
import torch

DATA_CSV = "/kg_0225.csv"                 
WORKDIR  = "/KG"                  
os.makedirs(WORKDIR, exist_ok=True)

TRIPLES_TSV = os.path.join(WORKDIR, "kg_triples.tsv")


MODEL_NAME = "RotatE"
EMBED_DIM  = 256
EPOCHS     = 50
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading CSV and generating triples ...")
df = pd.read_csv(DATA_CSV, low_memory=False)


triples = df[['x_id', 'relation', 'y_id']].dropna().drop_duplicates()


triples[['x_id', 'relation', 'y_id']].to_csv(TRIPLES_TSV, sep='\t', header=False, index=False)
print(f"Triples saved to {TRIPLES_TSV}  (#triples={len(triples)})")


tf: TriplesFactory = TriplesFactory.from_path(
    TRIPLES_TSV,
    create_inverse_triples=True,
)


train_tf, valid_tf, test_tf = tf.split(
    ratios=(0.8, 0.1, 0.1),
    random_state=SEED
)

print(f"#entities={tf.num_entities}, #relations={tf.num_relations}")
print(f"train={train_tf.num_triples}, valid={valid_tf.num_triples}, test={test_tf.num_triples}")


result = pipeline(
    model=MODEL_NAME,
    model_kwargs=dict(embedding_dim=EMBED_DIM),
    training=train_tf,
    validation=valid_tf,
    testing=test_tf,
    optimizer="Adam",
    optimizer_kwargs=dict(lr=LEARNING_RATE),
    negative_sampler="basic",
    training_kwargs=dict(
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    ),
    evaluator=RankBasedEvaluator(),
    stopper="early",
    stopper_kwargs=dict(
        frequency=5,
        patience=5,
        relative_delta=0.001,
    ),
    device=DEVICE,
    random_seed=SEED,
)

print("Training finished.")


ent_emb = result.model.entity_representations[0]  # Embedding
ent_tensor = ent_emb(indices=None).detach().cpu().numpy()  
entity_id_to_label = {idx: lbl for lbl, idx in result.training.entity_to_id.items()}

entity_index = np.arange(ent_tensor.shape[0])
entity_label = [entity_id_to_label.get(i, f"__missing__{i}") for i in entity_index]

ent_df = pd.DataFrame(ent_tensor)
ent_df.insert(0, "entity_id", entity_index)
ent_df.insert(1, "entity_label", entity_label)

# 
rel_emb = result.model.relation_representations[0]
rel_tensor = rel_emb(indices=None).detach().cpu().numpy()
relation_id_to_label = {idx: lbl for lbl, idx in result.training.relation_to_id.items()}
relation_index = np.arange(rel_tensor.shape[0])
relation_label = [relation_id_to_label.get(i, f"__missing__{i}") for i in relation_index]

rel_df = pd.DataFrame(rel_tensor)
rel_df.insert(0, "relation_id", relation_index)
rel_df.insert(1, "relation_label", relation_label)


ent_parquet = os.path.join(WORKDIR, "entity_embeddings.parquet")
rel_parquet = os.path.join(WORKDIR, "relation_embeddings.parquet")
ent_df.to_parquet(ent_parquet, index=False)
rel_df.to_parquet(rel_parquet, index=False)


pd.DataFrame(
    [(k, v) for v, k in result.training.entity_to_id.items()],
    columns=["entity_label", "entity_id"]
).sort_values("entity_id").to_csv(os.path.join(WORKDIR, "entity_id_map.tsv"), sep='\t', index=False)

pd.DataFrame(
    [(k, v) for v, k in result.training.relation_to_id.items()],
    columns=["relation_label", "relation_id"]
).sort_values("relation_id").to_csv(os.path.join(WORKDIR, "relation_id_map.tsv"), sep='\t', index=False)


metrics = result.get_metric_results().to_dict()
print("Eval (filtered ranking) metrics:", metrics)

with open(os.path.join(WORKDIR, "metrics.txt"), "w") as f:
    f.write(str(metrics))

print("Done. Outputs saved under:", WORKDIR)
