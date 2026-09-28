# Rolling Ball USD asset preparation

Pinned task: ReflexBench `8bb931485093c6d98f8729774ad01bf824964e16`, task `RollingBallInterception`.

The CPU dependency gate passes for the three USD roots used by the task. The remote mirror is `/root/autodl-tmp/rolling-ball-lewm/assets/local/Assets/Isaac/5.1`; it contains 22 files totaling 20,985,474 bytes. `UsdUtils.ComputeAllDependencies` (isolated `usd-core` 26.8) found every referenced USD layer and PNG file. For each root, its only unresolved non-file reference is the bare Kit material `OmniPBR.mdl`.

| Config property | Original USD URI | Mirrored path below asset root |
| --- | --- | --- |
| `cfg.scene.catcher.spawn.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Mugs/SM_Mug_A2.usd` | `Isaac/Props/Mugs/SM_Mug_A2.usd` |
| `cfg.scene.robot.spawn.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd` | `Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd` |
| `cfg.scene.plane.spawn.usd_path` | `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd` | `Isaac/Environments/Grid/default_environment.usd` |

The mirrored dependencies are:

- Mug: `Isaac/Props/Mugs/texture/T_Mug_A2_Albedo.png`, `T_Mug_A2_Normal.png`, and `T_Mug_A2_ORM.png`.
- Franka: `Isaac/IsaacLab/Robots/FrankaEmika/Materials/Materials.usd` and the 12 files under `Isaac/IsaacLab/Robots/FrankaEmika/Props/`: `instanceable_collision_meshes.usd`, `panda_hand.usd`, `panda_leftfinger.usd`, `panda_rightfinger.usd`, and `panda_link0.usd` through `panda_link7.usd`.
- Ground plane: `Isaac/Environments/Grid/Materials/Textures/WireframeBlur_basecolor.png`, `WireframeBlur_blue.png`, and `Wireframe_blue.png`.

The Normal texture was resumed from its retained partial using the equivalent dotted S3 hostname `omniverse-content-production.s3.us-west-2.amazonaws.com`; the object key and bucket are unchanged. Its final size is 1,899,700 bytes. The other files used the original dashed hostname.

USD dependency results: Mug `1` USD layer + `3` file assets; Franka `14` USD layers; ground plane `1` USD layer + `3` file assets. All checked local file references exist. The SDK kernel wheel contains the production MDL source member `isaacsim/kit/mdl/core/Base/OmniPBR.mdl`; the isolated USD parser cannot resolve Kit's MDL search path. Kit material resolution/compilation, rendering, and the native environment remain unverified until the rented-GPU smoke test.

The evaluator's `--asset-mirror` path mapping should substitute only these three `usd_path` values at `gym.make`; no task physics or pinned source changes are needed. No `ISAACSIM_ASSET_ROOT` setting was used. The task source and defaults are documented in [the pinned joint config](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/rolling_ball_interception/config/franka/joint_pos_env_cfg.py), [the pinned base config](https://github.com/LxRoboticsLab/ReflexBench/blob/8bb931485093c6d98f8729774ad01bf824964e16/source/reflexbench/reflexbench/tasks/manager_based/rolling_ball_interception/rolling_ball_interception_env_cfg.py), [IsaacLab v2.3.1 Franka config](https://github.com/isaac-sim/IsaacLab/blob/v2.3.1/source/isaaclab_assets/isaaclab_assets/robots/franka.py), and [IsaacLab v2.3.1 GroundPlane config](https://github.com/isaac-sim/IsaacLab/blob/v2.3.1/source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py).
