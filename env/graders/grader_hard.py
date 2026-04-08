from env.models import Action, Reward

FLAW_KEYWORDS = {
    "mujoco_single_seed": ["single seed", "seed=42", "fixed seed", "not multiple seeds", "one seed"],
    "atari_insufficient_episodes": ["10 episodes", "insufficient episodes", "not 100 episodes", "too few episodes"],
    "poor_transfer_hidden_in_appendix": ["transfer fails", "40% drop", "poor generalisation", "appendix hides", "buried in appendix"],
    "generalisation_claim_contradicted_by_results": ["contradicts results", "claim not supported", "generalisation fails", "results contradict"],
    "glue_difference_not_statistically_significant": ["p=0.07", "not significant", "p > 0.05", "no statistical significance"],
    "theoretical_assumption_not_validated_on_target_domains": ["assumption not validated", "power-law not tested", "assumption holds only on wikipedia"],
    "speed_benefit_only_at_long_sequences_not_highlighted": ["only at n>2048", "not highlighted", "short sequence", "512 tokens", "benefit not visible"],
    "overgeneralised_applicability_claim": ["overgeneralised", "too broad", "not all domains", "limited to"],
}

RED_HERRING_KEYWORDS = {
    "detailed_compute_disclosure": ["compute disclosed", "hardware disclosed", "tpu disclosed"],
    "code_release_promised": ["code released", "code available", "reproducible"],
    "ablation_study_present": ["ablation present", "ablation conducted"],
    "multiple_statistical_tests": ["wilcoxon", "statistical test", "significance test"],
    "formal_theorem_with_proof": ["theorem proven", "formal proof", "theoretical guarantee"],
    "multiple_baselines": ["multiple baselines", "many baselines"],
    "tpu_compute_disclosed": ["tpu disclosed"],
    "flops_methodology_stated": ["flops counted", "flops methodology"],
}


class HardGrader:
    def grade(self, action: Action, paper: dict, turn: int) -> Reward:
        planted_flaws = paper["planted_flaws"]
        red_herrings = paper["red_herrings"]
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

        # 3. Red herring penalty — if agent cites red herrings as decisive flaws, penalise
        red_herrings_cited = sum(
            1 for rh in red_herrings
            if any(kw in combined_text for kw in RED_HERRING_KEYWORDS.get(rh, [rh.replace("_", " ")]))
        )
        red_herring_penalty = min(red_herrings_cited * 0.1, 0.3)

        # 4. Multi-turn discovery bonus — reward agents that find more flaws in later turns
        turn_bonus = 0.0
        if turn >= 3 and flaws_found >= len(planted_flaws) * 0.75:
            turn_bonus = 0.1

        # 5. Justification quality
        justification_score = min(len(action.justification.split()) / 100, 1.0)

        # 6. Efficiency — correct and confident
        efficiency_bonus = 0.05 if decision_score == 1.0 and action.confidence >= 0.7 else 0.0

        total = max(
            0.0,
            0.30 * decision_score
            + 0.35 * flaw_score
            + 0.15 * justification_score
            + 0.10 * turn_bonus
            + 0.05 * efficiency_bonus
            - red_herring_penalty,
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
                "red_herrings_cited": red_herrings_cited,
                "red_herring_penalty": red_herring_penalty,
                "turn_bonus": turn_bonus,
                "turn": turn,
            },
        )
