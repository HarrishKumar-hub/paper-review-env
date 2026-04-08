from env.models import Action, Reward

KNOWN_FLAW_KEYWORDS = {
    "no_baseline_comparison": ["no baseline", "no comparison", "without baseline", "missing baseline"],
    "single_dataset_overgeneralisation": ["single dataset", "only one dataset", "not generalise", "overfit", "overgeneralised"],
    "no_ablation_study": ["no ablation", "ablation missing", "without ablation"],
    "no_confidence_intervals": ["no confidence interval", "no std", "no standard deviation", "no error bar", "single run"],
    "single_seed_evaluation": ["single seed", "one seed", "seed not varied", "no multiple seeds"],
    "unfair_baseline_evaluation": ["unfair baseline", "different prompt", "not fair comparison", "biased evaluation"],
    "no_code_release": ["no code", "code not released", "not reproducible", "evaluation code missing"],
    "cherry_picked_comparison": ["cherry picked", "cherry-picked", "selective comparison", "biased comparison"],
    "implausible_claim": ["implausible", "impossible", "cannot beat", "unrealistic", "extraordinary claim"],
}


class EasyGrader:
    def grade(self, action: Action, paper: dict) -> Reward:
        ground_truth_decision = paper["ground_truth_decision"]
        planted_flaws = paper["planted_flaws"]

        # 1. Decision correctness (0 or 1)
        decision_score = 1.0 if action.decision == ground_truth_decision else 0.0

        # 2. Flaw detection (how many planted flaws were identified)
        identified_text = " ".join(action.identified_flaws).lower() + " " + action.justification.lower()
        flaws_found = 0
        for flaw in planted_flaws:
            keywords = KNOWN_FLAW_KEYWORDS.get(flaw, [flaw.replace("_", " ")])
            if any(kw in identified_text for kw in keywords):
                flaws_found += 1

        flaw_score = flaws_found / max(len(planted_flaws), 1)

        # 3. Justification quality — length and specificity check
        justification_score = min(len(action.justification.split()) / 50, 1.0)

        # 4. Efficiency bonus — high confidence + correct decision
        efficiency_bonus = 0.1 if decision_score == 1.0 and action.confidence >= 0.8 else 0.0

        total = (
            0.4 * decision_score
            + 0.35 * flaw_score
            + 0.15 * justification_score
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
                "justification_words": len(action.justification.split()),
            },
        )
