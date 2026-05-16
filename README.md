# image-to-embedding

Convierte imágenes en vectores de **512 dimensiones** usando un backbone ResNet18 preentrenado y los almacena en PostgreSQL con la extensión **pgvector**.

## Requisitos previos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Python | 3.13 | |
| [uv](https://docs.astral.sh/uv/) | cualquiera | gestor de paquetes y entornos |
| PostgreSQL | 13+ | con la extensión `pgvector` instalada |
| CUDA *(opcional)* | 11.8+ | solo si quieres aceleración GPU |

### Instalar uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Instalar la extensión pgvector en PostgreSQL

```sql
-- como superusuario en psql
CREATE EXTENSION IF NOT EXISTS vector;
```

> Si usas Docker, la imagen `pgvector/pgvector:pg16` ya incluye la extensión.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd image-to-embedding

# 2. Instalar dependencias (uv crea el .venv automáticamente)
uv sync
```

`uv sync` descarga torch, torchvision, pillow, psycopg2 y pgvector en un entorno virtual aislado en `.venv/`. No necesitas activarlo manualmente; usa `uv run` para ejecutar scripts dentro de él.

---

## Configuración de la base de datos

Exporta la cadena de conexión como variable de entorno antes de correr el script:

```bash
export DATABASE_URL="postgresql://usuario:contraseña@localhost:5432/nombre_bd"
```

O pásala directamente con `--dsn` en cada llamada.

El script crea la tabla automáticamente en la primera ejecución:

```sql
CREATE TABLE image_embeddings (
    id         SERIAL PRIMARY KEY,
    filename   TEXT NOT NULL,
    embedding  vector(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## Uso

### Sintaxis general

```
uv run main.py [IMAGEN ...] [opciones]
```

### Opciones

| Opción | Predeterminado | Descripción |
|---|---|---|
| `--dsn` | `$DATABASE_URL` | Cadena de conexión PostgreSQL |
| `--table` | `image_embeddings` | Nombre de la tabla destino |
| `--device` | `cuda` si hay GPU, si no `cpu` | Dispositivo PyTorch |
| `--no-db` | — | Imprime el embedding en stdout sin conectarse a la BD |

### Ejemplos

```bash
# Probar sin base de datos (imprime los primeros 4 valores del vector)
uv run main.py foto.jpg --no-db

# Procesar una imagen e insertar en la BD
uv run main.py foto.jpg

# Procesar un directorio completo de imágenes
uv run main.py imagenes/*.jpg imagenes/*.png

# Especificar conexión y tabla manualmente
uv run main.py foto.jpg \
    --dsn "postgresql://user:pass@localhost:5432/mydb" \
    --table mis_embeddings

# Forzar CPU aunque haya GPU disponible
uv run main.py foto.jpg --device cpu
```

### Salida esperada

```
[OK] foto.jpg → row id 1
[OK] perro.png → row id 2
```

Con `--no-db`:

```
foto.jpg: [0.231, -0.104, 0.873, 0.012] ... (512 dims)
```

---

## Consultas de ejemplo con pgvector

Una vez almacenados los embeddings puedes hacer búsquedas por similitud:

```sql
-- Las 5 imágenes más similares a la imagen con id = 1 (distancia coseno)
SELECT id, filename, 1 - (embedding <=> (SELECT embedding FROM image_embeddings WHERE id = 1)) AS similitud
FROM image_embeddings
WHERE id != 1
ORDER BY embedding <=> (SELECT embedding FROM image_embeddings WHERE id = 1)
LIMIT 5;

-- Índice HNSW para búsquedas aproximadas eficientes en datasets grandes
CREATE INDEX ON image_embeddings USING hnsw (embedding vector_cosine_ops);
```

---

## Arquitectura del modelo

```
Imagen (cualquier tamaño)
    │
    ▼  Resize(256) → CenterCrop(224) → Normalize (ImageNet)
    │
    ▼  ResNet18 preentrenado (pesos ImageNet) — sin capa FC final
    │
    ▼  Average Pooling global
    │
    ▼  Vector de 512 dimensiones  →  PostgreSQL (pgvector)
```

El backbone ResNet18 se descarga automáticamente de PyTorch Hub la primera vez que se ejecuta el script (~45 MB).

---

## Estructura del proyecto

```
image-to-embedding/
├── main.py          # script principal
├── pyproject.toml   # dependencias y metadatos del proyecto
├── uv.lock          # versiones exactas de dependencias
└── .venv/           # entorno virtual (generado por uv, no commitear)
```

---

## Dependencias principales

| Paquete | Versión | Rol |
|---|---|---|
| `torch` | ≥ 2.12 | motor de inferencia |
| `torchvision` | ≥ 0.27 | modelos y transformaciones de imagen |
| `pillow` | ≥ 12.2 | carga de imágenes |
| `psycopg2-binary` | ≥ 2.9 | driver PostgreSQL |
| `pgvector` | ≥ 0.4 | soporte del tipo `vector` en psycopg2 |
