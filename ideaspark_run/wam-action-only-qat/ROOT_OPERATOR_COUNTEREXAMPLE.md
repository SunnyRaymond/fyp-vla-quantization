# Suffix-response loss versus original endpoint fidelity

Root analytic check, 2026-10-03. This is a scalar operator example, not a simulated learned model, a robot experiment, or a structural impossibility verdict. All choices below are illustrative. The independent technical reviewer should assess its consequence for the proposal's claims.

Take two solver transitions, identity action decoding and a one-dimensional executed prefix. Both BF16 transitions are identity. The quantized student's first transition has a fixed offset b, and its second has an adjustable offset c. Thus the BF16 original endpoint is z, while the student original endpoint is z+b+c.

At k=0, the matched teacher/student remaining-suffix endpoint difference is b+c. At k=1, the student-visited state is z+b; the BF16 identity suffix ends at z+b, while the student suffix ends at z+b+c. Their difference is c. Uniformly sampling these two indices gives

    L_suffix(c) = ((b+c)^2 + c^2) / 2.

For fixed nonzero b, the loss is minimized at c=-b/2. Starting from c=-b, however, the original student endpoint exactly matches BF16. Moving to c=-b/2 reduces suffix loss from b^2/2 to b^2/4 while increasing the original endpoint error from zero to b/2. For instance, b=1 and c in {-1,-0.5,0} gives this example even on a finite scalar grid.

Consequences:

- Large student-visited suffix mismatch can coexist with exact original endpoint fidelity. Internal path mismatch alone does not establish a control defect.
- Reducing the proposed suffix objective need not improve original action endpoint fidelity when earlier errors are constrained and later errors compensate for them. This is a target/optimization tradeoff, rather than merely a missing experiment.
- The proposed closed-loop association remains empirical. It must be evaluated against ordinary original-endpoint KD and student-state per-step OPD under matched training cost; changing only teacher endpoint pairing does not isolate this tradeoff.
- The example does not show that every WAM has a fixed offset b, that QAT cannot improve control, or that the candidate fails for all admissible parameters. It identifies a condition the method rationale or limitations must confront. Root has not added an anchor loss, changed the candidate, or prescribed a redesign.
