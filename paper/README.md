# PAMPAr-Coder V3 — Paper & Registration

## Archivos

| Archivo               | Propósito                                       |
| --------------------- | ----------------------------------------------- |
| `pampar_v3_arxiv.tex` | Paper LaTeX (formato arXiv, cs.CL / cs.LG)      |
| `../CITATION.cff`     | Metadata CFF para GitHub "Cite this repository" |
| `../.zenodo.json`     | Metadata para registro DOI en Zenodo            |
| `../LICENSE`          | Business Source License 1.1 (BUSL-1.1)          |

## Compilar el paper

```bash
# Requiere: texlive-full o miktex
pdflatex pampar_v3_arxiv.tex
bibtex pampar_v3_arxiv
pdflatex pampar_v3_arxiv.tex
pdflatex pampar_v3_arxiv.tex
```

O con `latexmk`:

```bash
latexmk -pdf pampar_v3_arxiv.tex
```

## Registrar en Zenodo

1. Ir a [zenodo.org](https://zenodo.org) → Login con GitHub
2. Settings → GitHub → Enable el repo `lucasmella-stack/PAMPAr-Coder`
3. Crear un **GitHub Release** (tag `v3.0.0`)
4. Zenodo detecta automáticamente el release y crea el DOI
5. Actualizar el DOI en `pampar_v3_arxiv.tex` (línea `\date`) y `CITATION.cff`

### Release tag sugerido

```bash
git tag -a v3.0.0 -m "PAMPAr-Coder V3: 2D Stream Architecture with Mixed Selectivity"
git push origin v3.0.0
```

## Subir a arXiv (si se consigue endorsement)

arXiv requiere endorsement para primeras submissions en cs.CL/cs.LG.
Opciones:

- Pedir endorsement a un autor que ya haya publicado en esa categoría
- Publicar primero en Zenodo (DOI valido) y academia.edu
- Usar el DOI de Zenodo como referencia oficial mientras tanto

### Preparar submission arXiv

```bash
# Crear zip con .tex + figuras
zip arxiv_submission.zip pampar_v3_arxiv.tex
```

## Subir a academia.edu

1. Login en academia.edu
2. Upload paper → PDF compilado de `pampar_v3_arxiv.tex`
3. Tags: code generation, brain-inspired AI, mixed selectivity, FiLM, curriculum learning
4. Vincular DOI de Zenodo

## Licencia

**Business Source License 1.1 (BUSL-1.1)**

- Uso no comercial (investigación, educación, experimentación): **libre**
- Uso comercial en producción: requiere licencia comercial
- **Change Date: 7 abril 2030** → se convierte a Apache 2.0 automáticamente
- Contacto licencias comerciales: lucas.mella@outlook.com
