from pathlib import Path
import unittest


class ArchitectureTests(unittest.TestCase):
    def test_docling_imports_are_isolated_to_infrastructure(self):
        checked_roots = [Path("parser_core/domain"), Path("parser_core/application")]
        offenders = []
        for root in checked_roots:
            for path in root.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                if "import docling" in source or "from docling" in source:
                    offenders.append(str(path))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
