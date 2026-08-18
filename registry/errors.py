class RegistryError(Exception):
    def __init__(self, message: str, code: str = "REGISTRY_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, message={self.message})"


class RegistryNoMatchError(RegistryError):
    def __init__(self, message: str = "No matching definitions found"):
        super().__init__(message, "REGISTRY_NO_MATCH")


class RegistryVersionInvalidError(RegistryError):
    def __init__(self, message: str = "Registry version is invalid or not published"):
        super().__init__(message, "REGISTRY_VERSION_INVALID")


class RegistryImmutableError(RegistryError):
    def __init__(self, message: str = "Cannot modify a published (immutable) definition"):
        super().__init__(message, "REGISTRY_IMMUTABLE")


class RegistryValidationError(RegistryError):
    def __init__(self, message: str = "Definition failed validation"):
        super().__init__(message, "REGISTRY_VALIDATION_ERROR")
