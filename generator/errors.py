class GeneratorError(Exception):
    def __init__(self, message: str, code: str = "GENERATOR_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, message={self.message})"


class SuiteIncompleteError(GeneratorError):
    """Raised when a TestSuite is persisted but has fewer cases than planned."""

    def __init__(self, suite_id: str, generated: int, expected: int, reasons: list):
        super().__init__(
            f"Suite '{suite_id}' incomplete: {generated}/{expected} cases generated.",
            "SUITE_INCOMPLETE",
        )
        self.suite_id = suite_id
        self.generated = generated
        self.expected = expected
        self.reasons = reasons


class GeneratorUnavailableError(GeneratorError):
    def __init__(self, message: str = "Agentic generator is unavailable"):
        super().__init__(message, "GEN_UNAVAILABLE")


class GeneratorSeedError(GeneratorError):
    def __init__(self, message: str = "No seeds available for generation"):
        super().__init__(message, "GEN_NO_SEEDS")


class GeneratorNotFoundError(GeneratorError):
    def __init__(self, message: str = "Generator artifact not found"):
        super().__init__(message, "GEN_NOT_FOUND")
