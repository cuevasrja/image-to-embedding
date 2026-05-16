"""
Convert images to 512-dimensional embeddings using a pretrained ResNet18 backbone
and store them in PostgreSQL via pgvector.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from pgvector.psycopg2 import register_vector
from PIL import Image

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_encoder(device: torch.device) -> nn.Module:
    """ResNet18 truncated before the final classifier → 512-d output."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    encoder = nn.Sequential(*list(backbone.children())[:-1])  # drop FC layer
    encoder.eval()
    return encoder.to(device)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def image_to_embedding(image_path: str | Path, encoder: nn.Module, device: torch.device) -> list[float]:
    """Return a 512-d embedding for a single image file."""
    img = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)      # (1, 3, 224, 224)

    with torch.no_grad():
        features = encoder(tensor)                        # (1, 512, 1, 1)

    return features.squeeze().cpu().tolist()              # [float x 512]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_connection(dsn: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    return conn


def setup_table(conn: psycopg2.extensions.connection, table: str = "image_embeddings") -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        SERIAL PRIMARY KEY,
                filename  TEXT NOT NULL,
                embedding vector(512) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
    conn.commit()


def insert_embedding(
    conn: psycopg2.extensions.connection,
    filename: str,
    embedding: list[float],
    table: str = "image_embeddings",
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"INSERT INTO {table} (filename, embedding) VALUES (%s, %s) RETURNING id;",
            (filename, embedding),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed images and store in pgvector.")
    p.add_argument("images", nargs="+", metavar="IMAGE", help="Image file paths")
    p.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/embeddings"),
        help="PostgreSQL connection string (or set DATABASE_URL env var)",
    )
    p.add_argument("--table", default="image_embeddings", help="Target table name")
    p.add_argument("--no-db", action="store_true", help="Print embeddings to stdout, skip DB")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    encoder = build_encoder(device)

    conn = None
    if not args.no_db:
        conn = get_connection(args.dsn)
        setup_table(conn, args.table)

    for path in args.images:
        path = Path(path)
        if not path.is_file():
            print(f"[WARN] {path} not found — skipping", file=sys.stderr)
            continue

        embedding = image_to_embedding(path, encoder, device)
        assert len(embedding) == 512, f"Expected 512 dims, got {len(embedding)}"

        if conn:
            row_id = insert_embedding(conn, path.name, embedding, args.table)
            print(f"[OK] {path.name} → row id {row_id}")
        else:
            print(f"{path.name}: {embedding[:4]} ... ({len(embedding)} dims)")

    if conn:
        conn.close()


if __name__ == "__main__":
    main()
