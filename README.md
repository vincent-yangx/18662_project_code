# 用来记录 版本更新，下一步规划 和 当前困惑的：

## 更改历史及版本更新：
资源收集完后立刻出现在共享仓库
地图上每种资源出现多次
同一位置的资源只能采集一次
添加仓库
目前设定为tool是全局共享的(Done)，
修改采矿的prerequisite(已完成)
目前是地图上有一个固定的仓库位置，agent走到仓库上就会把所有东西放进仓库
设立一个可取的图标(已完成， 是 exit)

## 下一步规划：
- LLM 传什么进去


Haven't Started: 接入LLM 的相关细节：
- 如何存储记忆（用何种数据结构）， 如何根据过往记忆进行行动路径的优化
- 给LLM传什么
- 是否要变成局部可知的
- 当地图上的资源不足以支持当前agent完成任务时，他会不会从仓库中取资源，或者要求其他agent先将资源放入仓库再去取
- communication cost怎么定义（上面关于资源的考量可以作为其中一部分）
- 执行任务中是否可以打断


## 当前困惑：

如果给agent设定了目标，在他运动过程中可以进行commuicate（更新目标）或者打断他吗

Memory 在 里面怎么进行存储， 怎么调用

是否要将每个agent的属性设置不同，以确保specialization

后续可以考虑工具的随身携带不可共享性(Haven't started)

agent的背包容量


