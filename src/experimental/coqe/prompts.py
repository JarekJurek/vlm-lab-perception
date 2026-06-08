from __future__ import annotations

from typing import Any

from datasets.coqe import COQERecord, COQETaskExample


PROMPT_VERSION = "coqe_qwen_v1"

SHARED_INSTRUCTION = """You are evaluating a visual reasoning task about liquid containers.
Use only the visual information in the image and the highlighted query container(s).
Do not use metadata, annotations, filenames, or ground-truth labels; answer only from the provided image and highlighted query container(s).
Think about object size, scene context, container shape, transparency, visible liquid level, and available free space.
Return only valid JSON in the specified schema.
Do not include explanations."""


def build_prompt(example: COQETaskExample) -> str:
    if example.task == "volume":
        return _join(SHARED_INSTRUCTION, VOLUME_PROMPT)
    if example.task == "content":
        return _join(SHARED_INSTRUCTION, CONTENT_PROMPT)
    if example.task == "comparative":
        if example.target is None:
            raise ValueError("Comparative examples require a target record.")
        return _join(SHARED_INSTRUCTION, _comparative_prompt(example.source, example.target))
    if example.task == "pouring":
        if example.angle is None:
            raise ValueError("Pouring examples require an angle.")
        return _join(SHARED_INSTRUCTION, POURING_PROMPT.replace("[ANGLE_DEGREES]", str(int(example.angle))))
    raise ValueError(f"Unknown COQE task: {example.task!r}")


VOLUME_PROMPT = """Task: Estimate the total capacity of the highlighted container.

The highlighted container is inside the yellow bounding box. Estimate the volume of the container when it is completely full, not the amount of liquid currently inside it.

Use visual scale cues from the whole image, including surrounding objects, container shape, and scene context. Choose the closest volume bin.

Allowed answers:
- "0_50_ml" for 0 mL <= volume < 50 mL
- "50_100_ml" for 50 mL <= volume < 100 mL
- "100_200_ml" for 100 mL <= volume < 200 mL
- "200_300_ml" for 200 mL <= volume < 300 mL
- "300_500_ml" for 300 mL <= volume < 500 mL
- "500_750_ml" for 500 mL <= volume < 750 mL
- "750_1000_ml" for 750 mL <= volume < 1000 mL
- "1000_2000_ml" for 1000 mL <= volume < 2000 mL
- "2000_3000_ml" for 2000 mL <= volume < 3000 mL
- "3000_inf_ml" for 3000 mL <= volume

Return only:
{"volume_bin": "<one_allowed_answer>"}"""


CONTENT_PROMPT = """Task: Estimate how full the highlighted container is.

The highlighted container is inside the yellow bounding box. Estimate the fraction of the container's total capacity that is currently filled with liquid.

Use visible cues such as the liquid surface line, color change, transparency, reflections, and container opacity. If the container is opaque and the amount of liquid inside cannot be visually estimated, choose "opaque".

Allowed answers:
- "0_percent" for empty
- "33_percent" for about one third full
- "50_percent" for about half full
- "66_percent" for about two thirds full
- "100_percent" for full or almost full
- "opaque" if the liquid amount cannot be visually determined because the container is opaque

Return only:
{"fill_level": "<one_allowed_answer>"}"""


POURING_PROMPT = """Task: Predict how much liquid will remain in the highlighted container over time if it is tilted.

The highlighted container is inside the yellow bounding box.
Tilt angle: [ANGLE_DEGREES] degrees.

Assume the container starts upright and is gradually tilted from 0 degrees to [ANGLE_DEGREES] degrees. Predict the fraction of liquid remaining inside the container at each of exactly five timesteps, corresponding to t0_content, t1_content, t2_content, t3_content, and t4_content.

Use the visible initial fill level, container geometry, opening shape, neck width, and tilt angle. A narrow opening may retain liquid longer. A wide opening may spill earlier.

Allowed sequence values:
- 0.0
- 0.1
- 0.2
- 0.3
- 0.4
- 0.5
- 0.6
- 0.7
- 0.8
- 0.9
- 1.0
- "opaque"

Return exactly five values.
If the container is initially empty, return [0.0, 0.0, 0.0, 0.0, 0.0].
If the container is opaque, return ["opaque", "opaque", "opaque", "opaque", "opaque"].

Return only:
{"remaining_sequence": [<five_allowed_values>]}"""


def _comparative_prompt(source: COQERecord, target: COQERecord) -> str:
    del source, target
    return """Task: Determine whether the entire liquid content of the source container can be poured into the target container without overflowing.

The source container is inside the yellow bounding box.
The target container is inside the green bounding box.

Estimate:
1. how much liquid is currently in the source container,
2. how much free capacity remains in the target container,
3. whether the source liquid can fit into the target container.

Choose "yes" only if all liquid currently in the source container can fit into the remaining empty space of the target container.
Choose "no" if it would overflow.
Choose "cant_tell" if at least one relevant container is opaque or the liquid amount cannot be visually estimated.

Allowed answers:
- "yes"
- "no"
- "cant_tell"

Return only:
{"can_pour": "<one_allowed_answer>"}"""


def _join(*parts: Any) -> str:
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())
