# Anotaciones

Única carpeta de `data/` que se versiona. Una subcarpeta por episodio:

```
data/annotations/<episodio>/events.json   # lista de Event (schemas.py)
```

- Los `src_spk`/`dst_spk` usan IDs anónimos `SPK_xx`. `scripts/evaluate.py` no los compara
  (dependen de la diarización): empareja por tipo y tiempo.
- `evidence` debe citar el texto literal que justifica la etiqueta.
- Nada de nombres de personas ni audio.
