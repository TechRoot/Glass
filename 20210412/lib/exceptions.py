"""
Excepciones personalizadas para gestionar dependencias externas ausentes.

Este módulo define una excepción utilizada para indicar que un
componente de la biblioteca requiere un controlador, servicio o módulo
externo que no está disponible en el entorno actual.  Se emplea para
que los llamadores distingan entre errores de lógica internos y
ausencia de dependencias sin revelar detalles de la implementación.
"""


class ExternalDependencyMissing(RuntimeError):
    """Módulo externo no incluido."""

    pass