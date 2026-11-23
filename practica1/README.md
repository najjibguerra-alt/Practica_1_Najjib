# Práctica 1 — Clasificador binario de business cases

PWA estática que clasifica un fragmento de texto en:

- **Finanzas / riesgo:** inversión, ingresos, evaluación financiera, riesgos.
- **Mercado / propuesta:** cliente, mercado, competencia, segmentación, solución,
  vehículo, ventas, marketing, operaciones, suministro, organización.

El modelo es TF-IDF de palabras + MLP de una capa oculta, entrenado desde cero con
fragmentos derivados de los documentos locales. No usa LLM ni un clasificador
preentrenado. La inferencia corre en el navegador; el texto no se envía a un servidor.

## Estado

- Corpus: 44 PDF de competencia + 2 PPTX, 46 documentos, 34 equipos.
- Dataset base: 666 fragmentos.
- Split temporal agrupado por equipo: train 500 (334 base + 166 extra), val 130, test 202.
- Test pendiente de auditoría manual al 100% (`data/labels_review.csv`).
- Modelo actual: provisional, porque usa etiquetas débiles en test para la evaluación.
- PWA funcional en local; falta publicar en GitHub Pages y probar la instalación en iPhone.

### Refuerzo de entrenamiento

Para llegar a 500 ejemplos de entrenamiento sin tocar val/test, `split.py` añade
166 fragmentos más finos (260–350 caracteres) de documentos cuyos equipos ya están
en train. Se deduplican contra todo el dataset y se muestrean con semilla fija.
Val y test conservan exactamente sus 130 y 202 filas.

### Corrida provisional (no usar como resultado final)

- Umbral calibrado: 0.26.
- Test débil: accuracy 0.762, precision 0.320, recall 0.533, F1 0.400.
- Costo esperado: 514.85 por 1000 casos.
- Baseline mayoría: 742.57 por 1000.
- Baseline keyword: 465.35 por 1000.
- Baseline logistic: 505.0 por 1000.
- ECE: 0.151 antes de calibrar, 0.084 después.

El baseline keyword es fuerte en esta corrida provisional y supera a la MLP en costo.
No se usa para elegir el umbral ni para reentrenar; se reporta como diagnóstico. La
decisión final se toma después de auditar el test al 100% y repetir la evaluación.

## Estructura

```text
practica1/
  PLAN.md                 Plan de ejecución y criterios de aceptación
  src/
    config.py             Rutas, secciones, costos, hiperparámetros
    text.py               PII, tokenización, chunking
    build_dataset.py      analysis.json -> fragments.csv
    split.py              split temporal + labels_review.csv
    audit.py              valida labels_review.csv -> test_labels.csv
    train_eval.py         TF-IDF + MLP, calibración, umbral, exportación
  data/                   Datos locales (no se publican)
  artifacts/              Métricas y modelo (local)
  notebooks/              Notebook de la rúbrica con dataset de train
  pwa/                    App estática publicable
  ood_cases.json          10 casos OOD sintéticos
  tests/                  Pruebas de datos e inferencia
```

## Pipeline

Desde la raíz del repositorio:

```bash
# 1. Construir fragmentos anonimizados
.venv/bin/python -m practica1.src.build_dataset

# 2. Crear split temporal y labels_review.csv
.venv/bin/python -m practica1.src.split

# 3. Etiquetar a mano los 202 fragmentos de test en
#    practica1/data/labels_review.csv (manual_label = 0 o 1)

# 4. Validar la auditoría
.venv/bin/python -m practica1.src.audit --check

# 5. Crear test_labels.csv
.venv/bin/python -m practica1.src.audit

# 6. Corrida final (sin --provisional)
.venv/bin/python -m practica1.src.train_eval
```

Para desarrollo, antes de la auditoría:

```bash
.venv/bin/python -m practica1.src.train_eval --provisional
```

## Evaluación

El clasificador no muestra el análisis en la página principal. El botón
**Análisis de errores** abre `evaluacion.html` en otra ventana. Ahí, errores y
fuera de dominio tienen scroll propio.

El notebook `notebooks/practica1_rubrica.ipynb` relaciona la app con la rúbrica
e incluye las 334 filas con las que se entrenó el modelo.

## Instalar

En la computadora basta abrir la URL local. En iPhone, Safari solo instala una PWA
por HTTPS:

1. Publica `practica1/pwa/` con GitHub Pages (workflow `.github/workflows/pages.yml`).
2. Abre la URL `https://<usuario>.github.io/<repo>/` en Safari.
3. Compartir → Añadir a pantalla de inicio.

`http://127.0.0.1:8087/` no se puede instalar en el iPhone.

## PWA en local

```bash
python3 -m http.server 8087 --bind 127.0.0.1 --directory practica1/pwa
```

Abrir `http://127.0.0.1:8087/`.

## Publicación en GitHub Pages

El workflow `.github/workflows/pages.yml` publica únicamente `practica1/pwa/`.
No publica `data/`, `artifacts/`, documentos originales ni etiquetas manuales.

Pasos:

1. Crear un repositorio público en GitHub.
2. Agregar el remoto y subir el proyecto.
3. En GitHub: Settings → Pages → Source: GitHub Actions.
4. Ejecutar el workflow `Deploy PWA`.
5. Abrir la URL HTTPS en Safari del iPhone.
6. Compartir → Añadir a pantalla de inicio.

La instalación PWA requiere HTTPS. `localhost` sirve para desarrollo; una IP local
sin HTTPS no permite el service worker en Safari.

## Privacidad

- `data/fragments.csv` y `data/labels_review.csv` son locales y están ignorados por git.
- El texto pasa por un scrub de correos, URLs, teléfonos, equipos, universidades y autores.
- La PWA pública solo contiene el modelo, el shell y ejemplos sintéticos.
- No se incluyen nombres, correos, teléfonos ni identificadores de personas.

## Limitaciones

- Las etiquetas de train/val son débiles (provienen de headings/secciones).
- El test se audita al 100% antes de la corrida final.
- El modelo puede fallar con OCR, tablas, secciones mixtas o textos muy cortos.
- La app rechaza entradas fuera de dominio en lugar de forzar una clase.
