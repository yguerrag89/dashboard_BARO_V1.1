# Dashboard Entradas Almacén (Streamlit)

Dashboard para visualizar el histórico de entradas diarias al almacén (producción), separando **PRIMERA** vs **SEGUNDA** y generando vistas útiles para Producción y Dirección.

---

## Reglas de negocio (actuales)

- **PRIMERA**: si el texto del producto contiene **BISA** o **IMU** o **POP**
- **SEGUNDA**: si contiene **SANSON**
- **OTROS**: todo lo demás (por ahora)

> Nota: Si en algún caso aparece SANSON y BISA juntos, se clasifica como **SEGUNDA** (prioridad a SANSON).

---

## Estructura del proyecto
├─ app.py
├─ core/
│ ├─ config.py
│ ├─ io_excel.py
│ ├─ rules.py
│ ├─ transform.py
│ └─ charts.py
├─ pages/
│ ├─ 01_Resumen_Ejecutivo.py
│ ├─ 02_Produccion_Diaria.py
│ ├─ 03_Calidad.py
│ ├─ 04_SKU_Detail.py
│ └─ 05_Alertas.py
├─ data/
│ ├─ input/
│ └─ curated/
├─ requirements.txt
└─ README.md



## Cómo correr
1) Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
2) Coloca tu Excel en `data/input/`.
3) (Opcional recomendado) Genera curado:
   ```
   python scripts/build_curated.py
   ```
4) Ejecuta:
   ```
   streamlit run app.py
   ```




