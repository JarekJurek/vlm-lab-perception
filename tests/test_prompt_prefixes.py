from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from tlg import PromptBank, make_tlg_annotation


class PromptPrefixesTest(unittest.TestCase):
    def test_task_prompt_prefix_is_prepended_to_annotation_question(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            prompt_defs = root / "prompt_bank.yaml"
            assignments = root / "dataset_prompts.yaml"
            prompt_defs.write_text(
                """
shared_context:
  question: |
    Shared context.

task_a:
  task_type: "Task A"
  question: |
    Task question.
""",
                encoding="utf-8",
            )
            assignments.write_text(
                """
example:
  include:
    - task_a
  task_prompt_prefixes:
    task_a:
      - shared_context
""",
                encoding="utf-8",
            )

            bank = PromptBank.from_files(prompt_defs, assignments)
            ann = make_tlg_annotation(
                bank,
                dataset="example",
                task="task_a",
                image_id="image",
                answer="answer",
            )

            self.assertEqual(ann.question, "Shared context.\n\nTask question.\n")

    def test_ot2_pipette_prefix_is_task_specific_and_neutral(self) -> None:
        bank = PromptBank.default()

        prefix = bank.task_prompt_prefix("ot2", "fill_percentage")

        self.assertIn("multiple pipette tips", prefix)
        self.assertNotIn("OT2", prefix)
        self.assertNotIn("ot2", prefix.lower())
        self.assertEqual(bank.task_prompt_prefix("ot2", "fill_level_ordinal"), prefix)
        self.assertEqual(bank.task_prompt_prefix("ot2", "liquid_grounding"), "")
        self.assertEqual(bank.task_prompt_prefix("ot2", "container_count"), "")

    def test_target_region_bbox_prompt_uses_generic_prefix_lookup(self) -> None:
        bank = PromptBank.default()

        self.assertEqual(
            bank.target_region_bbox_prompt(),
            bank.prompt_prefix("target_region_bbox_prompt"),
        )

    def test_dtld_target_region_prefix_is_task_specific(self) -> None:
        bank = PromptBank.default()

        self.assertEqual(
            bank.task_prompt_prefix("dtld", "container_type"),
            bank.target_region_bbox_prompt().rstrip(),
        )
        self.assertEqual(
            bank.task_prompt_prefix_ids("dtld", "container_type"),
            ["target_region_bbox_prompt"],
        )
        self.assertEqual(bank.task_prompt_prefix("dtld", "container_count"), "")

    def test_dtld_target_region_prefix_is_not_baked_into_annotation(self) -> None:
        bank = PromptBank.default()

        ann = make_tlg_annotation(
            bank,
            dataset="dtld",
            task="container_type",
            image_id="image",
            answer="T25 Flask",
        )

        self.assertFalse(ann.question.startswith("The target region is inside the red bounding box."))


if __name__ == "__main__":
    unittest.main()
