import unittest

from metrics import aggregate_sensitivity, endpoint_metrics, sensitivity_metrics, summarize


def blank_sample():
    return {"action": [[0.0] * 7 for _ in range(32)], "commands": [[0.0] * 7 for _ in range(32)]}


def copied(sample):
    return {key: [row[:] for row in value] for key, value in sample.items()}


def set_gripper(sample, signs):
    for step, sign in enumerate(signs):
        sample["action"][step][6] = float(sign)
        sample["commands"][step][6] = float(sign)


def synthetic_sensitivity(errors, side_error, flip_count=1):
    comparison = [
        {"dimension": dim, "rms": 0.5, "teacher_rms": 1.0, "teacher_error_rmse": errors[dim]}
        for dim in range(6)
    ]
    metric = {
        "central_directional_derivative": {"teacher_comparison_per_dimension": comparison},
        "one_sided_response": {
            side: {
                "motor_per_dimension": [
                    {"dimension": dim, "teacher_error_rmse": side_error[side][dim]}
                    for dim in range(6)
                ],
                "gripper_flip_count": flip_count,
                "teacher_gripper_flip_count": flip_count + 1,
                "gripper_flip_pattern_mismatch_steps": [1, 3][:flip_count],
            }
            for side in ("plus", "minus")
        },
    }
    return metric


class MetricsTests(unittest.TestCase):
    def test_endpoint_known_motor_errors_and_actual_gripper_signs(self):
        teacher = blank_sample()
        student = blank_sample()
        for step in range(10):
            student["action"][step][0] = 1.0
            student["commands"][step][0] = 3.0
        student["action"][0][1] = 2.0
        student["commands"][0][1] = -4.0

        student_signs = [-1, -1, 0, 0, 1, 1, -1, -1, -1, 1]
        teacher_signs = [-1, 0, 0, 0, 1, 1, 1, 1, -1, -1]
        set_gripper(student, student_signs)
        set_gripper(teacher, teacher_signs)

        result = endpoint_metrics(student, teacher)
        action_dims = result["motor"]["normalized_action"]["per_dimension"]
        command_dims = result["motor"]["executed_command"]["per_dimension"]
        self.assertEqual(action_dims[0]["rmse"], 1.0)
        self.assertEqual(action_dims[0]["mae"], 1.0)
        self.assertEqual(action_dims[0]["signed_bias"], 1.0)
        self.assertEqual(action_dims[0]["max_abs"], 1.0)
        self.assertAlmostEqual(action_dims[1]["rmse"], (4.0 / 10) ** 0.5)
        self.assertAlmostEqual(command_dims[0]["rmse"], 3.0)
        self.assertEqual(command_dims[1]["max_abs"], 4.0)

        grip = result["gripper"]
        self.assertAlmostEqual(grip["normalized_rmse"], (13.0 / 10) ** 0.5)
        self.assertEqual(grip["command_sign_disagreement"], {
            "count": 4, "rate": 0.4, "steps": [1, 6, 7, 9]
        })
        self.assertEqual(grip["command_sign_confusion"]["matrix"], [[2, 0, 1], [1, 2, 0], [2, 0, 2]])
        switches = grip["first10_switches"]
        self.assertEqual(switches["student_indices"], [2, 4, 6, 9])
        self.assertEqual(switches["teacher_indices"], [1, 4, 8])
        self.assertEqual(switches["timing_mismatch_steps"], [1, 2, 6, 8, 9])

    def test_linear_directional_derivative_and_one_sided_response(self):
        epsilon = 0.001
        base = blank_sample()
        plus = copied(base)
        minus = copied(base)
        for step in range(10):
            slope = float(2 * (step + 1))
            plus["action"][step][0] = epsilon * slope
            minus["action"][step][0] = -epsilon * slope
        result = sensitivity_metrics(base, plus, minus, base, plus, minus, epsilon)
        central = result["central_directional_derivative"]
        self.assertAlmostEqual(central["student_per_dimension"][0]["mean"], 11.0)
        self.assertEqual(central["teacher_comparison_per_dimension"][0]["teacher_error_rmse"], 0.0)
        self.assertEqual(central["teacher_comparison_per_dimension"][0]["teacher_relative_error"], 0.0)
        side = result["one_sided_response"]["plus"]["motor_per_dimension"][0]
        self.assertAlmostEqual(side["rms"], epsilon * (sum((2 * (i + 1)) ** 2 for i in range(10)) / 10) ** 0.5)
        self.assertEqual(side["teacher_error_rmse"], 0.0)

    def test_constant_student_detects_nonzero_teacher_response(self):
        epsilon = 1.0 / 255.0
        base = blank_sample()
        student_plus = copied(base)
        student_minus = copied(base)
        teacher_plus = copied(base)
        teacher_minus = copied(base)
        for step in range(10):
            teacher_plus["action"][step][2] = 2 * epsilon
            teacher_minus["action"][step][2] = -2 * epsilon

        result = sensitivity_metrics(
            base, student_plus, student_minus,
            base, teacher_plus, teacher_minus,
            epsilon,
        )
        compared = result["central_directional_derivative"]["teacher_comparison_per_dimension"][2]
        self.assertEqual(compared["rms"], 0.0)
        self.assertAlmostEqual(compared["teacher_rms"], 2.0)
        self.assertAlmostEqual(compared["teacher_error_rmse"], 2.0)
        self.assertAlmostEqual(compared["teacher_relative_error"], 1.0)
        self.assertEqual(result["one_sided_response"]["plus"]["motor_per_dimension"][2]["teacher_error_rmse"], 2 * epsilon)

        row = {
            "case_id": "2001", "suite": "pusht", "arm": "clip_only", "family": "image",
            "epsilon": epsilon, "normalized": result, "executed_command": result,
            "geometry": {"student_rms": 0.0, "teacher_rms": 2.0, "error_rms": 2.0, "gain": None, "cosine": None},
        }
        aggregate = aggregate_sensitivity([row])
        dim = aggregate["families"]["image"]["overall"]["arms"]["clip_only"]["central_directional_derivative"]["normalized"][2]
        self.assertEqual(dim["student_rms"], 0.0)
        self.assertAlmostEqual(dim["teacher_rms"], 2.0)
        self.assertAlmostEqual(dim["teacher_error_rmse"], 2.0)
        geometry = aggregate["families"]["image"]["overall"]["arms"]["clip_only"]["geometry"]
        self.assertEqual(geometry["error_rms"], {"mean": 2.0, "defined_case_count": 1})
        self.assertEqual(geometry["gain"], {"mean": None, "defined_case_count": 0})

    def test_aggregate_sensitivity_full_160_rows_and_pair_counts(self):
        rows = []
        suites = ("pusht", "cube", "reacher", "tworoom")
        errors = {
            "clip_only": [0.2, 0.1, 0.3, 0.4, 0.5, 0.6],
            "joint": [0.1, 0.2, 0.3, 0.4, 0.4, 0.7],
        }
        for case_index in range(40):
            case_id = str(2001 + 2 * case_index)
            suite = suites[case_index // 10]
            for family, epsilon in (("image", 1.0 / 255.0), ("proprio", 0.005)):
                for arm in ("clip_only", "joint"):
                    side_errors = {
                        "plus": [0.01 * (dim + 1) for dim in range(6)],
                        "minus": [0.02 * (dim + 1) for dim in range(6)],
                    }
                    metric = synthetic_sensitivity(errors[arm], side_errors, flip_count=1 if arm == "clip_only" else 2)
                    rows.append({
                        "case_id": case_id, "suite": suite, "arm": arm, "family": family, "epsilon": epsilon,
                        "normalized": metric, "executed_command": metric,
                        "geometry": {
                            "student_rms": 0.5, "teacher_rms": 1.0, "error_rms": 0.5,
                            "gain": None if case_index % 2 == 0 else 0.5, "cosine": 1.0,
                        },
                    })
        result = aggregate_sensitivity(rows)
        self.assertTrue(result["coverage"]["complete_160_row_panel"])
        self.assertEqual(result["coverage"]["row_count"], 160)
        image = result["families"]["image"]
        self.assertEqual(image["overall"]["case_count"], 40)
        self.assertEqual(image["by_suite"]["pusht"]["case_count"], 10)
        counts = image["overall"]["joint_vs_clip_normalized_derivative_error_by_dimension"]
        self.assertEqual(counts["improved"], [40, 0, 0, 0, 40, 0])
        self.assertEqual(counts["worsened"], [0, 40, 0, 0, 0, 40])
        self.assertEqual(counts["tied"], [0, 0, 40, 40, 0, 0])
        clip = image["overall"]["arms"]["clip_only"]
        self.assertEqual(clip["gripper"]["plus"]["flip_count"], {"total": 40, "case_mean": 1.0})
        self.assertEqual(clip["gripper"]["plus"]["flip_pattern_mismatch_steps"]["total"], 40)
        self.assertEqual(clip["geometry"]["gain"]["defined_case_count"], 20)
        self.assertEqual(clip["geometry"]["error_rms"]["defined_case_count"], 40)

        with self.assertRaises(ValueError):
            aggregate_sensitivity(rows + [rows[0]])
        incomplete = aggregate_sensitivity(rows[:-1])
        self.assertFalse(incomplete["coverage"]["complete_160_row_panel"])
        self.assertFalse(incomplete["families"]["proprio"]["coverage"]["complete_family"])

    def test_summary_keeps_five_variants_inside_one_case(self):
        records = []
        variants = ("base", "image_plus", "image_minus", "proprio_plus", "proprio_minus")
        suites = ("pusht", "cube", "reacher", "tworoom")
        for case_index in range(40):
            case_id = str(2001 + 2 * case_index)
            for variant in variants:
                for arm, magnitude in (("clip_only", 1.0), ("joint", 0.5), ("bf16", 0.0)):
                    sample = blank_sample()
                    for step in range(10):
                        for dim in range(6):
                            sample["action"][step][dim] = (
                                2.0 if arm == "clip_only" and case_index == 39 and dim == 0 else magnitude
                            )
                    records.append({
                        "case_id": case_id,
                        "suite": suites[case_index // 10],
                        "arm": arm,
                        "variant": variant,
                        "action": sample["action"],
                        "commands": sample["commands"],
                        "sampler_seed": 0,
                        "input_perturbation": {"variant": variant},
                    })
        result = summarize(records)
        self.assertEqual(result["coverage"]["case_count"], 40)
        self.assertEqual(result["coverage"]["record_count"], 600)
        self.assertEqual(len(result["case_metrics"]), 200)
        self.assertNotIn("by_case", result["endpoint"]["base"])
        self.assertEqual(result["endpoint"]["base"]["paired_case_count"], 40)
        self.assertEqual(len(result["endpoint"]["perturbed"]), 4)
        self.assertTrue(result["coverage"]["complete_40_case_4_suite_cohort"])
        self.assertTrue(result["coverage"]["complete_600_record_panel"])
        self.assertEqual(result["endpoint"]["base"]["by_suite_case_weighted"]["pusht"]["case_count"], 10)
        counts = result["endpoint"]["base"]["joint_vs_clip_by_motor_dimension"]
        self.assertEqual(counts["improved"], [40] * 6)
        self.assertEqual(counts["worsened"], [0] * 6)
        clip_dim0 = result["endpoint"]["base"]["overall_case_weighted"]["clip_only_endpoint_case_weighted"]["motor"]["normalized_action"][0]
        self.assertEqual(clip_dim0["max_abs"], 2.0)

    def test_shape_finite_and_epsilon_guards(self):
        sample = blank_sample()
        malformed = copied(sample)
        malformed["action"].pop()
        with self.assertRaises(ValueError):
            endpoint_metrics(malformed, sample)
        nonfinite = copied(sample)
        nonfinite["commands"][0][0] = float("inf")
        with self.assertRaises(ValueError):
            endpoint_metrics(nonfinite, sample)
        with self.assertRaises(ValueError):
            sensitivity_metrics(sample, sample, sample, sample, sample, sample, 0.0)


if __name__ == "__main__":
    unittest.main()
