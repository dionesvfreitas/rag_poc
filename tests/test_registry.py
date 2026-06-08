import unittest

from parser_core.registry import ParserRegistry


class RegistryTests(unittest.TestCase):
    def test_register_get_and_list(self):
        registry = ParserRegistry()
        parser = object()
        registry.register("docling", parser)

        self.assertIs(registry.get("docling"), parser)
        self.assertEqual(registry.list(), ["docling"])

    def test_missing_parser_raises_clear_error(self):
        registry = ParserRegistry()

        with self.assertRaises(KeyError):
            registry.get("missing")


if __name__ == "__main__":
    unittest.main()
