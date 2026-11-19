"""Disclosure attack protocol.

Screening distances are not membership-inference results. A reportable
attack scores the accumulated teacher requests and responses, uses a
disjoint calibration set, and returns 不可判定 when the sample is too small
or the score is not a likelihood ratio / shadow-model score.
"""

from __future__ import annotations

from dataclasses import dataclass

INCONCLUSIVE = "不可判定"
MIN_EVAL_PER_CLASS = 1000
MIN_EXPECTED_FP_AT_001 = 5
LOW_FPRS = (0.001, 0.01)


@dataclass(frozen=True)
class AttackerView:
    """What the attacker is allowed to see. Evaluator membership labels are separate."""

    sees_disclosed_requests: bool = True
    sees_disclosed_responses: bool = True
    sees_generator_weights: bool = False
    sees_sealed_test: bool = False
    sees_undisclosed_reference_rows: bool = False


@dataclass(frozen=True)
class AttackAssessment:
    status: str
    reasons: tuple[str, ...]
    kind: str
    fprs: tuple[float, ...] = LOW_FPRS
    tpr_at_fpr: dict[str, float] | None = None
    intervals: dict[str, tuple[float, float]] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "kind": self.kind,
            "fprs": list(self.fprs),
            "tpr_at_fpr": self.tpr_at_fpr,
            "intervals": self.intervals,
        }


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float] | None:
    """95% Clopper-Pearson interval, or None if SciPy is absent."""
    if n <= 0:
        return None
    try:
        from scipy.stats import beta
    except ImportError:
        return None
    if successes <= 0:
        lower = 0.0
    else:
        lower = float(beta.ppf(alpha / 2, successes, n - successes + 1))
    if successes >= n:
        upper = 1.0
    else:
        upper = float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lower, upper


def screening_nearest_distance(distance: float) -> dict[str, object]:
    """A nearest-neighbor screen. Not an attack result."""
    return {
        "kind": "screening",
        "distance": distance,
        "privacy_proof": False,
        "membership_inference": False,
    }


def assess_disclosed_attack(
    *,
    n_disclosed_requests: int,
    n_disclosed_with_response: int,
    n_member_eval: int,
    n_nonmember_eval: int,
    n_member_calibration: int,
    n_nonmember_calibration: int,
    calibration_eval_overlap: int,
    has_likelihood_ratio_score: bool,
    attacker: AttackerView | None = None,
) -> AttackAssessment:
    """Decide whether a strong attack on the disclosed log can be reported."""
    view = attacker or AttackerView()
    reasons: list[str] = []
    if view.sees_generator_weights or view.sees_sealed_test or view.sees_undisclosed_reference_rows:
        reasons.append("attacker view exceeds the disclosed request/response log")
    if not view.sees_disclosed_requests or not view.sees_disclosed_responses:
        reasons.append("attacker cannot see both disclosed requests and responses")
    if n_disclosed_requests <= 0 or n_disclosed_with_response <= 0:
        reasons.append("disclosed request/response log is empty")
    if n_disclosed_with_response < n_disclosed_requests:
        reasons.append("some disclosed requests have no stored response")
    if n_member_eval < MIN_EVAL_PER_CLASS:
        reasons.append(f"member evaluation n={n_member_eval} < {MIN_EVAL_PER_CLASS}")
    if n_nonmember_eval < MIN_EVAL_PER_CLASS:
        reasons.append(f"non-member evaluation n={n_nonmember_eval} < {MIN_EVAL_PER_CLASS}")
    expected_fp = n_nonmember_eval * 0.001
    if expected_fp < MIN_EXPECTED_FP_AT_001:
        reasons.append(
            f"expected false positives at FPR 0.1% is {expected_fp:.2f} < {MIN_EXPECTED_FP_AT_001}"
        )
    if n_member_calibration <= 0 or n_nonmember_calibration <= 0:
        reasons.append("calibration split is missing")
    if calibration_eval_overlap != 0:
        reasons.append("calibration and evaluation records overlap")
    if not has_likelihood_ratio_score:
        reasons.append("no likelihood-ratio or shadow-model score")
    if reasons:
        return AttackAssessment(
            status=INCONCLUSIVE,
            reasons=tuple(reasons),
            kind="strong_mia",
        )
    return AttackAssessment(
        status="ready",
        reasons=(),
        kind="strong_mia",
    )


def refuse_screening_as_attack(screen: dict[str, object]) -> None:
    if screen.get("kind") != "strong_mia":
        raise ValueError(
            "nearest-neighbor or simple membership scores cannot be reported as the attack"
        )
