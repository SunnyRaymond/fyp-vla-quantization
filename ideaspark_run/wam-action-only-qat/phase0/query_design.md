# Retrieval query design

The verbatim request and binding user steering are supplied to the orchestrator from REQUEST.md; it writes them to user_query.txt. OOD triggers do not fire: a quantization method scope, action-generating WAM family, explicit anchor papers, initial Fast-WAM platform, data and hardware constraints are present.

| Role | Query | Vocabulary ownership and concrete object |
|---|---|---|
| Broad domain | world action model action diffusion | World-action model and action diffusion are embodied generative-policy vocabulary; concrete object: generated action trajectory. |
| Method signature | quantization aware training diffusion robot action policy | Robot action policy anchors QAT to embodied diffusion rather than language-model quantization; concrete object: denoised robot action. The mixed ownership remains a deliberate low-yield probe for generic QAT competitors. |
| Most similar problem | video action joint generation shared backbone policy | Joint video-action generation belongs to WAM and generative robot-policy architectures; concrete objects: action tokens and video frames sharing a backbone. |
| Escape mechanism | diffusion policy quantization timestep error distillation | Diffusion policy owns the action denoising setting; timestep error compensation and distillation are plausible named solutions to accumulated quantization error. Concrete object: diffusion-policy denoising timestep/action trajectory. Generic distillation dominance is limited by the diffusion-policy phrase but remains a documented retrieval risk. |

Exactly four queries preserve each query's capped retrieval share. No fifth query is added. Named user anchors will be connector resolved; QVGen, LingBot-VA and Cosmos Policy need exact-title resolution rather than assumed paper identities. The search is a literature map, not a new mechanism proposal.
