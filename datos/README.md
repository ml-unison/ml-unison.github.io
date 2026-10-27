# Datos del curso

Las libretas cargan estos archivos con `rdp.datos.cargar(nombre)`. La función lee la copia local si existe y, si no, descarga la versión de GitHub. Guardamos copias en el repositorio para que las libretas sigan funcionando aunque la fuente original cambie o no esté disponible.

| Archivo | Contenido | Fuente | Licencia |
|---|---|---|---|
| `almac_reciente.xlsx` | Almacenamiento diario (hm³) de 10 presas, enero de 2020 a junio de 2023. Columnas `clave`, `fecha`, `almacenamiento`. | Extracto del conjunto [Recursos Hídricos](https://datos.sonora.gob.mx/dataset/Recursos%20H%C3%ADdricos) del portal de datos abiertos del Gobierno del Estado de Sonora, publicado por SAGARHPA. | CC BY |
| `catalogo_estatal.xlsx` | Catálogo de presas: nombre, río, municipio y capacidades (NAMO, NAME). | El mismo conjunto, recurso «Catálogo». | CC BY |
| `casas_ames.csv` | Ventas de casas en Ames, Iowa, 2006–2010: 1,460 casas y 79 atributos. | D. De Cock, *Journal of Statistics Education* 19(3), 2011. Versión de [OpenML 42165](https://www.openml.org/d/42165). | Ver la ficha de OpenML |
| `censo_adult.csv` | 12,000 personas del censo de EE. UU. de 1994, sacadas con `random_state=1994`. Los faltantes conservan el `?` original y la columna objetivo se llama `ingreso`. | R. Kohavi y B. Becker, UCI Machine Learning Repository. Versión de [OpenML 1590](https://www.openml.org/d/1590). | CC BY 4.0 |
| `tienda_facturas.csv` | 25,900 facturas de una tienda en línea del Reino Unido (dic. 2010 – dic. 2011), una fila por factura: `factura`, `cliente`, `fecha`, `lineas`, `piezas`, `importe` (libras), `pais`. Las facturas que empiezan con «C» son cancelaciones. | Agregado por factura del conjunto *Online Retail* de D. Chen, UCI Machine Learning Repository (2015), doi:10.24432/C5BW33. | CC BY 4.0 |
| `pinguinos.csv` | 344 pingüinos de tres especies del archipiélago Palmer, Antártida: medidas del pico, la aleta y la masa. | K. Gorman, Palmer Station LTER; paquete `palmerpenguins` de A. Horst, A. Hill y K. Gorman (2020). | CC0 |
| `titanic.csv` | Los 1,309 pasajeros del *RMS Titanic*. | T. Cason a partir de la *Encyclopedia Titanica*, distribuida por F. Harrell (Vanderbilt). Versión de OpenML. | Ver la ficha de OpenML |

El portal de Sonora tiene un archivo por década desde 1941. Si se quiere actualizar el extracto o ampliar el periodo, por ejemplo para el proyecto, hay que descargar los archivos a mano desde el navegador, porque el portal rechaza las descargas automáticas.
