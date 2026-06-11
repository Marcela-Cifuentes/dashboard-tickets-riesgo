"""
etl.py
======
Proceso de actualización programada (ejecutar cada 1 hora, FUERA de Streamlit).

Qué hace:
  1. Descarga cada base Excel desde Google Cloud Storage.
  2. Ejecuta el pipeline completo (limpieza + NLP + inferencia del modelo
     + anomalías) UNA sola vez.
  3. Persiste resultados en data/<base>.parquet (resultados intermedios).
  4. Escribe data/metadata.json con timestamp y conteos (el dashboard lo
     muestra como "última actualización").

El dashboard (app.py) únicamente LEE estos parquet: nunca reentrena ni
ejecuta inferencia masiva.

Programación según entorno (ver DESPLIEGUE.md):
  - Linux (cron):        0 * * * * cd /ruta/proyecto && python etl.py
  - Windows (schtasks):  Programador de tareas -> python etl.py cada hora
  - Streamlit Cloud:     GitHub Actions (.github/workflows/etl.yml)

Uso manual:
  python etl.py                 # procesa todas las bases
  python etl.py --base TicketsMintic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone

import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("etl")


def ejecutar(bases: list[str]) -> int:
    pipeline.DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict = {
        "actualizado_utc": datetime.now(timezone.utc).isoformat(),
        "bases": {},
    }
    errores = 0

    for base in bases:
        t0 = time.perf_counter()
        try:
            log.info("Procesando base '%s'...", base)
            df = pipeline.procesar_base(base)

            ruta = pipeline.DATA_DIR / f"{base}.parquet"
            df.to_parquet(ruta, index=False)

            dur = time.perf_counter() - t0
            metadata["bases"][base] = {
                "filas": int(len(df)),
                "columnas": int(df.shape[1]),
                "segundos_proceso": round(dur, 1),
                "estado": "ok",
            }
            log.info("OK '%s': %d filas -> %s (%.1fs)", base, len(df), ruta, dur)

        except Exception as exc:  # noqa: BLE001 - el ETL no debe morir por una base
            errores += 1
            metadata["bases"][base] = {"estado": f"error: {exc}"}
            log.exception("Falló la base '%s'", base)

    (pipeline.DATA_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Metadata escrita. Errores: %d", errores)
    return 1 if errores else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL HelpDesk -> parquet")
    parser.add_argument(
        "--base",
        choices=list(pipeline.URLS_BASES.keys()),
        help="Procesar solo una base (por defecto: todas)",
    )
    args = parser.parse_args()

    bases = [args.base] if args.base else list(pipeline.URLS_BASES.keys())
    sys.exit(ejecutar(bases))
