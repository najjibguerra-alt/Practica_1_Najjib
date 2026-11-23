# Práctica 1 — Plan de ejecución

## 1. Decisiones cerradas

| Tema | Decisión |
|---|---|
| Producto | PWA estática que clasifica un fragmento de texto. No se convierte el dashboard completo. |
| Hosting | GitHub Pages, publicando únicamente `practica1/pwa/`. |
| Etiquetas | Etiqueta débil desde secciones del dashboard para train; test etiquetado manual al 100%. |
| Split | Temporal, agrupado por equipo. Si el test no alcanza el mínimo de positivos, el script se detiene para revisión; no se fuerza un split no temporal. |
| Modelo | TF-IDF + MLP de una capa oculta, implementada con scikit-learn. Sin pesos preentrenados. |
| Costos | `C_FP = 1`, `C_FN = 5`. Sensibilidad reportada con 1:3 y 1:10. |
| Entrada | Solo texto pegado. No PDF, no OCR, no archivo. |
| Etiqueta positiva | `inversion`, `ingresos`, `financiera`, `riesgos`. |
| Etiqueta negativa | `cliente`, `mercado`, `competencia`, `segmentacion`, `solucion`, `vehiculo`, `ventas`, `marketing`, `operaciones`, `suministro`, `organizacion`. |
| Secciones excluidas | `portada`, `indice`, `resumen`, `conclusion`, `anexos`. |
| Secciones ambiguas excluidas | `empresa`, `mision`, `sustentabilidad`, `implementacion`, `posventa`. |
| Privacidad | Anonimizar correos, URLs, teléfonos, nombres de equipos y universidades; usar IDs `T01`/`D01`. |
| Entregables | Primero plan + código. PDF y video solo después de aprobación y verificación del usuario. |

## 2. Problema

Clasificador binario de fragmentos de business case:

- **Positivo:** contenido financiero o de riesgo.
- **Negativo:** contenido de mercado, propuesta u operación.

El modelo recibe únicamente texto. Las variables del dashboard se usan para construir etiquetas, agrupar el split y auditar; nunca como features.

## 3. Dataset

### Fuente

- `data/analysis.json`: documentos, páginas, líneas, fechas, secciones y headings.
- `data/text/*.txt`: respaldo textual.
- 46 documentos, 34 equipos, 401 páginas.

### Construcción de fragmentos

1. Segmentar por headings: cada heading con sección canónica abre un segmento que termina en el siguiente heading, aunque el siguiente no tenga sección. Esto evita mezclar “Inversión” con “Operaciones” en una misma página.
2. Si una página no tiene headings útiles, usar el fallback de `structure.section_hits`:
   - ventanas de páginas por sección;
   - elegir la sección con inicio más reciente que cubra la página;
   - si hay empate, prioridad positivo > negativo > ambiguo > excluido.
3. Excluir secciones excluidas o ambiguas.
4. Construir el texto desde `pages[].lines[]`, no desde el texto plano.
5. Eliminar headings detectados, números de página, footers repetidos, correos, URLs y teléfonos.
6. Fragmentar por oraciones en bloques de 350–650 caracteres, objetivo 450 para el dataset base; el refuerzo de train usa 260–350.
7. Descartar fragmentos con menos de 80 caracteres o con proporción de letras menor a 0.40.
8. Deduplicar por hash de texto normalizado.
9. Guardar `fragments.csv` con: `fragment_id`, `doc_key`, `team_key`, `section`, `label`, `text`, `char_len`, `page`.
10. Escribir `dataset_manifest.json` con conteos, distribución por clase y sección, fechas, hash del CSV y parámetros.

### Gate de aceptación

- `>= 500` fragmentos únicos.
- Al menos 15% de la clase positiva y al menos 30 positivos en test.
- Al menos 25 documentos y 15 equipos representados.
- Sin correos, URLs, teléfonos ni nombres de equipos/universidades.
- Manifiesto con hash.

Estado actual: **666 fragmentos base**, 46 documentos, 34 equipos; **138 positivos / 528 negativos**. El corpus incluye 44 PDF de competencia y 2 PPTX. El manifiesto y el hash están en `data/dataset_manifest.json`.

## 4. Split

1. Parsear `metadata.CreationDate`; si falta, `ModDate`.
2. Fecha del equipo = fecha máxima de sus documentos.
3. Ordenar equipos por fecha.
4. Test: equipos más recientes hasta cubrir ~20% de los fragmentos y al menos 30 positivos y 40 negativos.
5. Validación: siguientes equipos hasta ~15%.
6. Train: resto.
7. Equipos sin fecha: solo train; se documenta la limitación.
8. Congelar `split.json` con hashes, conteos y fecha.
9. Exportar `labels_review.csv` con todos los fragmentos de test y columnas `manual_label`, `notes`.

Estado actual: train **500** (115 pos / 385 neg; 166 fragmentos extra de train), val **130** (27 pos / 103 neg), test **202** (30 pos / 172 neg). El test tiene 13 equipos y 18 documentos. Los equipos sin fecha quedan en train. El refuerzo nunca toca val/test.

### Regla de oro

- El test no se usa para elegir modelo, features, calibración ni umbral.
- Si el test se toca antes de la corrida final, se reporta el resultado anterior y la corrección.

## 5. Modelo

### Pipeline

- TF-IDF de palabras, unigramas y bigramas.
- `min_df=2`, `max_features=12000`, `sublinear_tf=True`, `strip_accents="unicode"`.
- Ajuste del vectorizador solo con train.
- MLP de una capa oculta, 12 neuronas, activación ReLU, salida sigmoide.
- `solver="lbfgs"`, `alpha=1e-3`, `max_iter=1200`, `random_state=42`.
- El modelo es pequeño; se densifica la matriz TF-IDF solo para la MLP y se verifica que los pesos exportados sean finitos.
- Comparación obligatoria contra:
  - baseline mayoría;
  - baseline regla tonta con palabras financieras.

### Calibración

- Curva de fiabilidad y ECE con 10 bins.
- Temperature scaling ajustado en validación.
- Reportar ECE y costo antes y después.
- El test se evalúa una sola vez.

### Umbral

- Barrido de 0.01 a 0.99 en validación.
- `C(t) = C_FP * FP(t) + C_FN * FN(t)`.
- Costo esperado por 1000 casos.
- Sensibilidad con `C_FN = 3` y `C_FN = 10`.
- Umbral final exportado a la PWA.

## 6. PWA

### Pantallas

1. **Clasificar:** textarea, botón, resultado.
2. **Resultado:** clase, probabilidad, umbral, estado de revisión.
3. **Ayuda:** qué hace, límites y privacidad.
4. **OOD:** estado “no clasificable” con razón.

### Inferencia

- Cargar `model.json` una vez.
- Tokenizar en JavaScript con la misma regla que Python.
- Calcular TF-IDF, MLP y temperatura en el cliente.
- Responder en menos de 2 segundos tras la carga.
- Si el texto está vacío, es muy corto, no parece español o tiene demasiadas palabras fuera del vocabulario, rechazar.

### PWA

- `manifest.webmanifest`.
- Service worker con cache del shell y del modelo.
- Iconos 192 y 512.
- Instalación en iPhone vía Safari → Compartir → Añadir a pantalla de inicio.
- Sin backend, sin login, sin datos personales.

## 7. Evaluación

- Matriz de confusión en test.
- Precisión, recall, F1, ROC-AUC, PR-AUC.
- Costo esperado por 1000 casos.
- Baseline mayoría y baseline keyword con su costo.
- ECE y curva de fiabilidad.
- 20 errores más informativos agrupados por causa.
- 10 entradas OOD con respuesta real y esperada.
- Intervalos de confianza bootstrap agrupados por equipo.

## 8. Entregables posteriores

No se generan hasta que el usuario apruebe y verifique el código:

- Video de 3–5 minutos.
- PDF de 6–8 páginas.

## 9. Fases y tiempo estimado (10–20 horas)

| Fase | Horas | Resultado |
|---|---:|---|
| Dataset + split + manifiesto | 3–4 | `fragments.csv`, `split.json`, `labels_review.csv` |
| Modelo + evaluación provisional | 3–4 | `model.json`, `metrics.json` |
| PWA + inferencia JS | 3–4 | App funcionando en local |
| Auditoría manual del test | 2–3 | `test_labels.csv` completo |
| Corrida final + revisión | 2–3 | Métricas finales y test congelado |
| GitHub Pages + verificación iPhone | 1–2 | URL HTTPS funcionando |
| PDF/video | 4–6 | Solo después de aprobación |

## 10. Riesgos

| Riesgo | Mitigación |
|---|---|
| Menos de 500 fragmentos de train | Añadir fragmentos finos solo de documentos de train; si faltan, pedir más reportes PDF. |
| Test temporal con pocos positivos | Ampliar test a más equipos recientes; si no alcanza, detenerse y documentar. |
| Etiquetas débiles ruidosas | Auditar 100% del test; reportar acuerdo con la etiqueta débil. |
| MLP sobreajusta o no supera baseline | Capa oculta pequeña, alpha, comparación explícita con baseline keyword; documentar si el baseline gana. |
| Probabilidades descalibradas | Temperature scaling en validación; reportar ECE y costo. |
| PWA no instala | Publicar solo archivos estáticos en GitHub Pages con HTTPS; probar en Safari iPhone. |
| Inferencia JS no coincide con Python | Test de paridad con los mismos textos y comparación de probabilidades. |
| Datos personales | Scrub de PII antes de escribir `fragments.csv`; no publicar documentos originales. |
| Falta de tiempo | Priorizar dataset, modelo, PWA y evaluación; PDF/video al final. |

## 11. Criterios de aceptación final

- [ ] `>= 500` fragmentos etiquetados y manifiesto con hash.
- [ ] Test 100% etiquetado manualmente.
- [ ] Split temporal agrupado por equipo.
- [ ] Baseline con métrica y costo.
- [ ] Matriz de confusión y umbral por costo.
- [ ] Costo esperado por 1000 casos.
- [ ] ECE y corrección aplicada con efecto en costo.
- [ ] 20 errores agrupados.
- [ ] 10 OOD con respuesta esperada.
- [ ] PWA en HTTPS, instalable en iPhone.
- [ ] Inferencia < 2 s.
- [ ] Video de 3–5 min.
- [ ] PDF de 6–8 páginas.
- [ ] Sin datos personales.
- [ ] Sin LLM ni clasificador preentrenado.
- [ ] Test usado una sola vez.
