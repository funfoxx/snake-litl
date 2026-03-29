# Snake LITL

This is a fork of chynl's [Snake](https://github.com/chynl/snake) repo, adjusted to follow LITL

Test results (averaged over 1000 episodes):

| Solver | Average Length | Average Steps | Average Length / Average Steps | \(Average Length\)^2 / Average Steps |
| :----: | :------------: | :-----------: | :----------------------------: | :---------------------------------: |
|BENCHMARK Hamilton|**\[63.93\]**|717.83|0.089|**\[5.694\]**|
|BENCHMARK Greedy|60.15|904.56|0.066|3.100|
|BENCHMARK DQN|24.44|131.69|**\[0.186\]**|4.536|
|DQN, Human Metric|29.488|210.119|0.140|4.138|
|DQN, Any Metric|20.025|**\[116.35\]**|0.172|3.447|
|Any, Human Metric|59.894|633.483|0.095|5.663|
|Any, Any Metric|30.817|241.904|0.127|3.926|

