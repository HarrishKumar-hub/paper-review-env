from env.models import Action, Reward

FLAW_KEYWORDS = {
    "no_ood_evaluation": ["out-of-distribution", "ood", "external validation", "different domain", "generalisation not tested"],
    "hyperparams_tuned_on_val_set": ["validation set tuning", "val set tuning", "hyperparameter leak", "tuned on validation", "val set optimised"],
    "overgeneralised_applicability_claim": ["overgeneralised", "too broad", "not all tasks", "limited scope", "claim too strong"],
    "single_site_dataset": ["single site", "one hospital", "single source", "single institution"],
    "benefit_diminishes_with_more_data": ["diminishes", "not significant at 1000", "benefit reduces", "only low-data"],
}

STRENGTH_KEYWORDS = {
    "multi_seed_evaluation": ["multiple seeds", "multi-seed", "3 seeds", "5 seeds", "averaged over seeds"],
    "clear_baselines": ["baselines", "comparison", "baseline included"],
    "statistically_significant_results": ["p<0.05", "statistically significant", "statistical test"],
    "irb_approved": ["irb", "ethics approval", "institutional review"],
    "detailed_preprocessing": ["preprocessing", "data pipeline", "described in detail"],
    "multiple_data_regimes_tested": ["100 labels", "500 labels", "1000 labels", "multiple regimes"],
}


class MediumGrader:
    def grade(self, action: Action, paper: dict) -> Reward:
        planted_flaws = paper["planted_flaws"]
        genuine_strengths = paper["genuine_strengths"]
        ground_truth_decision = paper["ground_truth_decision"]

        combined_text = " ".join(action.identified_flaws).lower() + " " + action.justification.lower()
        if action.requested_changes:
            combined_text += " " + " ".join(action.requested_changes).lower()

        # 1. Decision correctness
        decision_score = 1.0 if action.decision == ground_truth_decision else 0.0

        # 2. Flaw detection
        flaws_found = sum(
            1 for flaw in planted_flaws
            if any(kw in combined_text for kw in FLAW_KEYWORDS.get(flaw, [flaw.replace("_", " ")]))
        )
        flaw_score = flaws_found / max(len(planted_flaws), 1)

        # 3. Strength acknowledgement — agent must not ignore genuine strengths
        strengths_acknowledged = sum(
            1 for strength in genuine_strengths
            if any(kw in combined_text for kw in STRENGTH_KEYWORDS.get(strength, [strength.replace("_", " ")]))
        )
        strength_score = strengths_acknowledged / max(len(genuine_strengths), 1)

        # 4. Justification quality
        justification_score = min(len(action.justification.split()) / 80, 1.0)

        # 5. Efficiency bonus
        efficiency_bonus = 0.1 if decision_score == 1.0 and action.confidence >= 0.75 else 0.0

        total = (
            0.35 * decision_score
            + 0.30 * flaw_score
            + 0.15 * strength_score
            + 0.10 * justification_score
            + 0.10 * efficiency_bonus
        )

        return Reward(
            total=round(total, 4),
            decision_score=decision_score,
            flaw_detection_score=round(flaw_score, 4),
            justification_score=round(justification_score, 4),
            efficiency_bonus=efficiency_bonus,
            breakdown={
                "flaws_found": flaws_found,
                "total_flaws": len(planted_flaws),
                "strengths_acknowledged": strengths_acknowledged,
                "total_strengths": len(genuine_strengths),
                "strength_score": round(strength_score, 4),
            },
        )
