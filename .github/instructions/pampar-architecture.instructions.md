# PAMPAr Architecture Instructions

> Arquitectura cerebral con 52 zonas de Brodmann para procesamiento de código.

## LLAVES System

LLAVES = Lookup de Activación Via Expresiones Sintácticas

```python
# Las LLAVES clasifican tokens usando patrones regex, NO ML
LLAVES = {
    'B06_KEYWORDS_IMPORT': ['import', 'from', 'require'],
    'B17_LITERAL_STRING': [r'".*"', r"'.*'", 'f"', "f'"],
    'B18_LITERAL_NUMERO': [r'\d+', r'\d+\.\d+'],
    'B35_IDENTIFICADOR_VAR': [r'[a-z_][a-z0-9_]*'],
    # ... 52 zonas total
}
```

## Territorios (4 Macro-Áreas)

| Territorio | Zonas | Función |
|------------|-------|---------|
| SINTAXIS | B01-B13 | Keywords, operadores, delimitadores |
| SEMÁNTICA | B14-B39 | Identificadores, literales, tipos |
| LÓGICO | B40-B44 | Condicionales, loops, excepciones |
| ESTRUCTURAL | B45-B52 | Patrones, estructuras, documentación |

## Flujo de Procesamiento

```
1. Token → LLAVES lookup (O(1), 80% peso)
2. Token → Embedding attention (30% peso)  
3. Combinar → Activación por zona
4. Zonas activas → Procesamiento territorial
5. Fusión → Output
```

## Cuantización

Solo las lookup tables de LLAVES se cuantizan a INT4:
- Reduce memoria de 6.5MB → 812KB
- Sin pérdida de precisión (es lookup discreto)
- El modelo (pesos) se mantiene en FP16/BF16

## Early Exit

El modelo puede salir temprano si la confianza es alta:
```python
if confianza > 0.90 and capa >= self.capas_minimas:
    return x, True  # Exit early
```

## Reglas de Implementación

1. **NUNCA** usar backprop para entrenar LLAVES
2. **SIEMPRE** procesar territorios en paralelo cuando sea posible
3. **Cuantizar** solo tablas de lookup, nunca pesos del modelo
4. **Registrar** tokenizer con `model.registrar_tokenizer(tokenizer)`
