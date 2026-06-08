from typing import Protocol


class Parser(Protocol):
    name: str

    def parse(self, source_path: str):
        ...


class ParserRegistry:
    def __init__(self):
        self._parsers = {}

    def register(self, name, parser):
        if not name:
            raise ValueError("Parser name is required.")
        self._parsers[name] = parser

    def get(self, name):
        try:
            return self._parsers[name]
        except KeyError as exc:
            available = ", ".join(self.list()) or "<none>"
            raise KeyError(f"Parser {name!r} is not registered. Available: {available}.") from exc

    def list(self):
        return sorted(self._parsers)
