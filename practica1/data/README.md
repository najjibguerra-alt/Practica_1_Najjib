# Datos locales de la Práctica 1

Este directorio contiene artefactos derivados de los documentos del repositorio:

- `fragments.csv`: fragmentos anonimizados con etiqueta débil.
- `fragments_split.csv`: fragmentos con split train/val/test. Train tiene 500 filas
  (334 base + 166 extra de documentos de train); val/test no se tocan.
- `labels_review.csv`: test para auditoría manual.
- `test_labels.csv`: etiquetas manuales finales.
- `dataset_manifest.json`, `split.json`: manifiestos reproducibles.

No se publican en GitHub Pages. El workflow solo publica `practica1/pwa/`.

Para regenerarlos:

```bash
.venv/bin/python -m practica1.src.build_dataset
.venv/bin/python -m practica1.src.split
```

Si se regenera el dataset, borra `labels_review.csv` y `test_labels.csv` para evitar
auditorías obsoletas.
