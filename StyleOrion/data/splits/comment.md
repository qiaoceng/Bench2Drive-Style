<!--
[Bench2Drive-Style] NEW FILE -- training-pickle generation
Author: YinXuan1223   (carried over from the ORION_NYCUACM_version fork)

Notes on the scene lists in this directory (bench2drive_our_select_mini.json,
bench2drive_our_select_whole.json, our_selected_scenarios.json), which decide
which Bench2Drive scenes prepare_B2D.py converts into the training pickle.
Records clip counts per scene, which scenes have BEV/multi-view renders, and
why individual scenes were kept or dropped. JSON cannot carry comments, so the
rationale lives here.
-->

### mini list (Ordering by pkl file):
| Scene Name                                               | Clip num | BEV + text + multi-view | 備註 |
|:-------------------------------------------------------- | -------- |:----------------------- | ---- |
| Accident_Town03_Route102_Weather20                       | 162      | V                       |      |
| NonSignalizedJunctionLeftTurn_Town03_Route122_Weather26  | 119      | V                       |      |
| SignalizedJunctionRightTurn_Town03_Route118_Weather14    | 111      | V                       |      |
| LaneChange_Town06_Route277_Weather9                      | 104      | V                       |      |
| NonSignalizedJunctionLeftTurn_Town03_Route123_Weather26  | 117      | V                       |      |
| MergerIntoSlowTrafficV2_Town12_Route1009_Weather21       | 112      | V                       |      |
| LaneChange_Town06_Route307_Weather21                     | 88       | V                       |      |
| VehicleTurningRoute_Town12_Route822_Weather18            | 491      | V                       |      |
| SignalizedJunctionLeftTurn_Town03_Route114_Weather6      | 105      | V                       |      |
| ConstructionObstacleTwoWays_Town12_Route1083_Weather9    | 315      | V                       |      |
| Accident_Town03_Route101_Weather23                       | 151      | V                       |      |
| MergerIntoSlowTraffic_Town06_Route317_Weather5           | 124      | V                       |      |
| MergerIntoSlowTraffic_Town12_Route1003_Weather8          | 123      | V                       | 自車撞到前車了...  |
| AccidentTwoWays_Town12_Route1102_Weather10               | 276      | V                       |      |
| HighwayExit_Town06_Route291_Weather5                     | 88       | V                       |      |
| ParkedObstacleTwoWays_Town12_Route1158_Weather14         | 310      | V                       |      |
| VehicleTurningRoute_Town12_Route1026_Weather12           | 201      | V                       |      |
| ConstructionObstacle_Town03_Route60_Weather8             | 186      | V                       |      |
| StaticCutIn_Town03_Route109_Weather1                     | 155      | V                       | 前車自己去撞柱子     |
| VehicleTurningRoutePedestrian_Town12_Route1040_Weather0  | 467      | V                       |      |
| HighwayCutIn_Town06_Route299_Weather13                   | 107      | V                       |      |
| HighwayExit_Town06_Route292_Weather14                    | 105      | V                       |      |
| VehicleTurningRoutePedestrian_Town12_Route1027_Weather13 | 178      | V                       |      |
| HighwayCutIn_Town06_Route298_Weather20                   | 108      | V                       |      |
| ParkedObstacle_Town03_Route147_Weather0                  | 155      | V                       | 這條路在隧道裡，所以比較不受天氣影響視覺     |
| ParkedObstacleTwoWays_Town12_Route1159_Weather23         | 299      | V                       |      |
| NonSignalizedJunctionRightTurn_Town03_Route126_Weather18 | 128      | V                       |      |
| ConstructionObstacle_Town03_Route61_Weather9             | 157      | V                       |      |
| ParkedObstacle_Town03_Route103_Weather25                 | 160      | V                       |      |
| NonSignalizedJunctionRightTurn_Town04_Route184_Weather2  | 109      | V                       |      |
| AccidentTwoWays_Town12_Route1103_Weather11               | 369      | V                       |      |
| StaticCutIn_Town03_Route110_Weather6                     | 158      | V                       |      |
| ConstructionObstacleTwoWays_Town12_Route1080_Weather14   | 158      | V                       |      |
| MergerIntoSlowTrafficV2_Town12_Route1010_Weather22       | 107      |                         |      |
| SignalizedJunctionRightTurn_Town03_Route151_Weather2     | 170      |                         |      |
| SignalizedJunctionLeftTurn_Town03_Route113_Weather26     | 125      |                         |      |

### weather description
可參考 https://carla.readthedocs.io/en/0.8.4/carla_settings/#weather-presets

選 1~8, 14, 18, 26

0. 👍白天
1. 👍白天
2. 👍白天
3. 👍白天 + 地上有水
4. ❌Wet cloudy noon，但沒找到 scene
5. 👍中雨
6. 👍下雨的白天
7. 👍白天。他說有下雨但我看不出來
8. 👍下雨的白天
9. ❔霧，看得到但稍微不清楚
10. 😵夜晚 + 雨
11. ❔霧，看得到但稍微不清楚
12. ❔霧
13. ❔霧
14. 👍小小雨。看起來還算清楚
15. ❔霧 + 大雨
16. ❌沒有 scene
17. ❌沒有 scene
18. 👍天色有點暗，但場景清楚
19. 😵夜晚
20. 😵夜晚
21. 😵夜晚 + 下雨
22. 😵夜晚
23. 😵夜晚 + 小雨。看得到但不清楚
24. ❌沒有 scene
25. 😵夜晚。這個超糟，幾乎看不到旁車
26. 👍白天 + 顏色很奇怪的天空

