"""Utilidades del curso Reconocimiento de Patrones (LCC · Universidad de Sonora).

En Colab:

    !pip install -q "rdp @ git+https://github.com/ml-unison/ml-unison.github.io@main#subdirectory=paquete"

    import rdp
    rdp.estilo.usar()              # gráficas con la paleta del curso
    rdp.quiz.mostrar("u01")        # cuestionario de autoevaluación
    df = rdp.datos.cargar("carretas")
"""

from . import datos, estilo, quiz

__all__ = ["datos", "estilo", "quiz"]
__version__ = "2027.1.0"
