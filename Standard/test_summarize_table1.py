from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd


class Table1SummaryTests(unittest.TestCase):
    def test_seeded_summary_reads_powershell_utf16_logs(self):
        tags = ("std", "bil", "noise", "nb", "bn")
        script = Path(__file__).with_name("summarize_table1.py")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for tag in tags:
                results = root / "Results" / f"Bil_10_{tag}_seed_42"
                results.mkdir(parents=True)
                pd.DataFrame([[0, 0.7]]).to_excel(results / "m.xlsx", index=False)
                pd.DataFrame([[0, 0.01]]).to_excel(results / "v.xlsx", index=False)

                for prefix, epsilon, values in (
                    ("mult", "0.03", "[0.1, 0.2, 0.3, 0.4, 0.5]"),
                    ("cw", "1", "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]"),
                ):
                    log = root / "logs" / f"{prefix}_{tag}_seed_42.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(f"Epsilon: {epsilon}\tTest Accuracy = {values}\n", encoding="utf-16")

            result = subprocess.run(
                [sys.executable, str(script), "--seed", "42"],
                cwd=root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = root / "table1_seed_42.md"
            self.assertTrue(report.is_file())
            self.assertIn("\u00b1", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
