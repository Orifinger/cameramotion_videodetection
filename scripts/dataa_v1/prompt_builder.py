"""Build VACE prompts from frozen case metadata only."""

from __future__ import annotations

from typing import Dict

from .schema import CanonicalCaseSpec


def _label(value: str | None, fallback: str) -> str:
    return value.strip() if value else fallback


def build_prompts(case: CanonicalCaseSpec) -> Dict[str, str]:
    target_name = _label(case.target.display_phrase or case.target.canonical_concept, "the target object")
    donor_name = _label(
        (case.donor.display_phrase or case.donor.canonical_concept) if case.donor else None,
        "the reference object",
    )
    domain_bits = [case.target.content_domain, case.target.style_domain]
    domain = ", ".join(bit for bit in domain_bits if bit)
    context = f" in a {domain} video" if domain else ""
    operation = case.operation or "local_edit"

    if operation == "object_swap":
        model = (
            f"The final frame shows {donor_name} naturally occupying the original location of {target_name}{context}, "
            "with matching scale, perspective, lighting, shadows, camera motion, and motion blur."
        )
    elif operation == "person_appearance_swap":
        model = (
            f"The person in the masked region has the appearance of {donor_name}{context}, while preserving the original "
            "pose, action, timing, camera motion, and scene geometry."
        )
    elif operation == "surface_content_edit":
        model = f"The masked surface contains coherent content matching {donor_name}{context}, aligned to the original plane and perspective."
    elif operation == "object_attribute_edit":
        model = f"The {target_name} keeps its position and motion{context}, but its visible material, color, or texture matches the requested edit."
    elif operation == "surface_attribute_edit":
        model = f"The masked surface keeps its geometry{context}, with updated color, coating, pattern, or material consistent with the scene."
    else:
        model = f"The masked local region around {target_name}{context} is edited coherently while the rest of the video remains unchanged."

    artifact_policy = case.sampling_meta.get("artifact_policy") or {}
    if artifact_policy.get("artifact_type") == "surface_text_degradation":
        model += " Text or fine markings inside the masked surface appear degraded, distorted, partially garbled, blurred, or visibly compressed while the surrounding scene remains coherent."

    control = (
        "Edit only the target mask tube and a reasonable boundary band. Preserve background, camera motion, pose, scene geometry, "
        "lighting, all non-edit regions, and temporal continuity. Donor RGB is reference-only and must not be copied into target compositing."
    )
    return {"model_prompt": model, "control_prompt": control}
