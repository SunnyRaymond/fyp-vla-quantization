"""KL on paired observation responses and frozen-reader motor outputs."""
from softbin_loss import pointwise_kl


def action_response_kl(torch, student_clean, student_perturbed,
                       teacher_clean, teacher_perturbed, stats):
    for value in (student_clean, student_perturbed, teacher_clean, teacher_perturbed):
        if tuple(value.shape) != (32, 7):
            raise ValueError('Expected full [32,7] actions; only first10 motor6 are supervised')
    student_delta = student_perturbed[:10, :6] - student_clean[:10, :6]
    teacher_delta = teacher_perturbed.detach()[:10, :6] - teacher_clean.detach()[:10, :6]
    # Use the original fixed action scale, never divide by a weak teacher response.
    return pointwise_kl(torch, student_delta, teacher_delta,
                        [0.0]*6, stats['action']['scale'], -1, stats['config'])


def latent_control_kl(torch, frozen_reader_action, teacher_action, stats):
    if tuple(frozen_reader_action.shape) != (32, 7) or tuple(teacher_action.shape) != (32, 7):
        raise ValueError('Expected [32,7] reader outputs')
    return pointwise_kl(torch, frozen_reader_action[:10, :6], teacher_action.detach()[:10, :6],
                        stats['action']['mean'], stats['action']['scale'], -1, stats['config'])
