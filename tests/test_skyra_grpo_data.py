from __future__ import annotations

import unittest

from scripts.skyra_grpo_diagnostics.build_datab_verl import (
    select_user_frame_lines,
    uniform_indices,
)


class SkyraDataConversionTests(unittest.TestCase):
    def test_uniform_17_to_16_is_unique_and_keeps_endpoints(self) -> None:
        indices = uniform_indices(17, 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(set(indices)), 16)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 16)

    def test_prompt_frame_line_selection_matches_images(self) -> None:
        text = "header\n" + "".join(f"[T={i}.00s] <image>\n" for i in range(17)) + "footer\n"
        selected = uniform_indices(17, 16)
        result = select_user_frame_lines(text, selected, 17)
        self.assertEqual(result.count("<image>"), 16)
        dropped = next(index for index in range(17) if index not in selected)
        self.assertNotIn(f"[T={dropped}.00s] <image>", result)


if __name__ == "__main__":
    unittest.main()
