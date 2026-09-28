"""Small PBS-side mechanism checks for the visual pilot model API."""

import torch

from visual_models import (encode_actions, make_encoder, make_predictor,
                           parameter_budget, parameter_count)


ARMS = ("full_gnn", "local", "local4", "global4", "global16",
        "transformer6", "flat_mlp_matched")


def _record(result, name, passed, message):
    result[name] = {"pass": bool(passed), "message": str(message)}


def run_model_checks(config, device="cpu"):
    """Run deterministic API and dependency checks inside the PBS allocation."""
    result = {}
    n, d, a = 5, 16, 4
    if hasattr(config, "get"):
        n = int(config.get("num_objects", config.get("representation", {}).get("num_slots", 5)))
        d = int(config.get("state_dim", config.get("representation", {}).get("slot_dim", 16)))
        a = int(config.get("action_dim", config.get("stage_a_encoder_fit", {}).get("action_dim_per_object", 4)))
    else:
        n, d, a = int(config.num_objects), int(config.state_dim), int(config.action_dim)

    try:
        routed = encode_actions(torch.tensor([0, 7, 19], device=device), n, a)
        expected = torch.zeros(3, n, a, device=device)
        expected[0, 0, 0] = 1
        expected[1, 1, 3] = 1
        expected[2, 4, 3] = 1
        passed = tuple(routed.shape) == (3, n, a) and torch.equal(routed, expected)
        _record(result, "action_target_routing", passed,
                "0->slot0/dir0, 7->slot1/dir3, 19->slot4/dir3; other slot actions are zero")
    except Exception as exc:
        _record(result, "action_target_routing", False, repr(exc))

    try:
        encoder = make_encoder(config, 1101).to(device).eval()
        image = torch.zeros(2, 3, 50, 50, device=device)
        slots = encoder(image)
        passed = tuple(slots.shape) == (2, n, d) and bool(torch.isfinite(slots).all())
        _record(result, "encoder_full_slot_output", passed,
                f"output shape={tuple(slots.shape)}; all values finite")
    except Exception as exc:
        _record(result, "encoder_full_slot_output", False, repr(exc))

    try:
        local4 = make_predictor("local4", config, 1101).to(device).eval()
        global4 = make_predictor("global4", config, 1101).to(device).eval()
        n_local, n_global = parameter_count(local4), parameter_count(global4)
        same_f = all(torch.equal(local4.model.local.state_dict()[k],
                                 global4.model.local.state_dict()[k])
                     for k in local4.model.local.state_dict())
        same_projection = torch.equal(
            local4.model.local_msg_weight.permute(1, 0, 2).reshape(4, n * d),
            global4.model.global_msg.weight)
        passed = (n_local == n_global and same_f and same_projection and
                  parameter_count(local4.model) - parameter_count(local4.model.local) == n * d * 4 and
                  parameter_count(global4.model) - parameter_count(global4.model.local) == n * d * 4)
        _record(result, "local4_global4_parameter_and_init_match", passed,
                f"local4={n_local}, global4={n_global}, shared local-function initialization={same_f}, shared projection initialization={same_projection}")

        state = torch.randn(2, n, d, device=device, requires_grad=True)
        actions = torch.tensor([0, 19], device=device)
        pred = local4(state, actions)
        grad = torch.autograd.grad(pred[:, 0].sum(), state)[0]
        cross = grad[:, 1:, :]
        passed = tuple(pred.shape) == (2, n, d) and torch.count_nonzero(cross).item() == 0
        _record(result, "local4_no_cross_slot_path", passed,
                "slot0 output has zero input-gradient to slots1..4")

        gstate = torch.randn(2, n, d, device=device, requires_grad=True)
        message = global4.model.global_msg(gstate.flatten(1))
        ggrad = torch.autograd.grad(message.sum(), gstate)[0]
        cross_count = int(torch.count_nonzero(ggrad[:, 1:, :]).item())
        _record(result, "global4_cross_slot_message_path", cross_count > 0,
                f"global summary depends on non-target slots; nonzero tested gradient entries={cross_count}")
    except Exception as exc:
        _record(result, "communication_structure", False, repr(exc))

    try:
        budget = parameter_budget(config)
        global4 = make_predictor("global4", config, 1101).to(device)
        flat = make_predictor("flat_mlp_matched", config, 1101).to(device)
        actual_global4 = parameter_count(global4)
        actual_flat = parameter_count(flat)
        relative = abs(actual_flat - actual_global4) / actual_global4
        passed = (relative <= 0.05 and actual_global4 == budget["global4_params"] and
                  actual_flat == budget["flat_params"])
        _record(result, "flat_mlp_parameter_budget", passed,
                f"actual global4={actual_global4}, actual flat={actual_flat}, "
                f"hidden={budget['flat_hidden']}, relative_difference={relative:.6f}")
    except Exception as exc:
        _record(result, "flat_mlp_parameter_budget", False, repr(exc))

    try:
        state = torch.randn(2, n, d, device=device)
        actions = torch.tensor([3, 16], device=device)
        shapes = {}
        for arm in ARMS:
            model = make_predictor(arm, config, 1101).to(device).eval()
            with torch.no_grad():
                pred = model(state, actions)
            shapes[arm] = tuple(pred.shape)
            if shapes[arm] != (2, n, d) or not bool(torch.isfinite(pred).all()):
                raise AssertionError(f"{arm}: shape={shapes[arm]}, finite={bool(torch.isfinite(pred).all())}")
            del model
        _record(result, "all_predictor_arms_api", True,
                f"all {len(ARMS)} arms return finite [B,slots,dim] next-state predictions")
    except Exception as exc:
        _record(result, "all_predictor_arms_api", False, repr(exc))

    return result
