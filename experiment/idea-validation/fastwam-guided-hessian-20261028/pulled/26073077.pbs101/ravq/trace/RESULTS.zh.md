# 夹爪归因：RAVQU_W2 对 QTIP_W2 的分歧回合

作业 `26073077.pbs101`，23 个分歧回合（RAVQU_W2 赢 12，QTIP_W2 赢 11）。重放复现原闭环成败：RAVQU_W2 23/23，QTIP_W2 23/23。

| 回合 | suite | 维度 | 赢家 | RAVQU_W2 结局 | RAVQU_W2 合/抖动 | QTIP_W2 结局 | QTIP_W2 合/抖动 |
|---|---|---|---|---|---|---|---|
| 15 | libero_spatial | camera_viewpoints | RAVQU_W2 | 成功 | 1/0 | 抓住后又掉 | 5/3 |
| 23 | libero_spatial | robot_initial_states | QTIP_W2 | 抓到又掉 | 2/0 | 成功 | 1/0 |
| 55 | libero_spatial | background_textures | RAVQU_W2 | 成功 | 1/0 | 抓到又掉 | 4/2 |
| 57 | libero_spatial | background_textures | QTIP_W2 | 抓住了但没完成 | 1/0 | 成功 | 1/0 |
| 80 | libero_object | camera_viewpoints | RAVQU_W2 | 成功 | 1/0 | 抓住了但没完成 | 1/0 |
| 84 | libero_object | camera_viewpoints | QTIP_W2 | 抓住了但没完成 | 7/5 | 成功 | 1/0 |
| 86 | libero_object | camera_viewpoints | RAVQU_W2 | 成功 | 2/1 | 抓到又掉 | 8/6 |
| 94 | libero_object | robot_initial_states | QTIP_W2 | 抓住后又掉 | 8/5 | 成功 | 3/2 |
| 117 | libero_object | light_conditions | RAVQU_W2 | 成功 | 1/0 | 抓到又掉 | 1/0 |
| 131 | libero_object | sensor_noise | RAVQU_W2 | 成功 | 1/0 | 抓住后又掉 | 11/6 |
| 150 | libero_goal | camera_viewpoints | QTIP_W2 | 抓住后又掉 | 6/3 | 成功 | 2/1 |
| 184 | libero_goal | light_conditions | QTIP_W2 | 始终没合 | 0/0 | 成功 | 0/0 |
| 188 | libero_goal | light_conditions | RAVQU_W2 | 成功 | 2/0 | 抓住后又掉 | 3/1 |
| 196 | libero_goal | background_textures | RAVQU_W2 | 成功 | 1/0 | 抓住了但没完成 | 3/3 |
| 199 | libero_goal | background_textures | RAVQU_W2 | 成功 | 1/0 | 抓到又掉 | 1/0 |
| 219 | libero_10 | objects_layout | QTIP_W2 | 抓住后又掉 | 10/7 | 成功 | 3/1 |
| 226 | libero_10 | camera_viewpoints | QTIP_W2 | 抓住后又掉 | 8/5 | 成功 | 4/2 |
| 229 | libero_10 | camera_viewpoints | QTIP_W2 | 抓到又掉 | 2/1 | 成功 | 3/1 |
| 235 | libero_10 | robot_initial_states | RAVQU_W2 | 成功 | 2/1 | 抓到又掉 | 1/0 |
| 237 | libero_10 | robot_initial_states | RAVQU_W2 | 成功 | 1/0 | 抓住了但没完成 | 1/0 |
| 252 | libero_10 | light_conditions | QTIP_W2 | 抓住了但没完成 | 13/11 | 成功 | 3/2 |
| 261 | libero_10 | background_textures | QTIP_W2 | 抓住后又掉 | 32/23 | 成功 | 3/1 |
| 274 | libero_10 | sensor_noise | RAVQU_W2 | 成功 | 1/0 | 始终没合 | 0/0 |

输家的结局：
- RAVQU_W2 输的 11 回合：抓住后又掉 5，抓住了但没完成 3，抓到又掉 2，始终没合 1（夹爪类失败 8）
- QTIP_W2 输的 12 回合：抓到又掉 5，抓住后又掉 3，抓住了但没完成 3，始终没合 1（夹爪类失败 9）
