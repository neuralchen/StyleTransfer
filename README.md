# StyleTransfer
 
## Usage
程序入口 main.py，此脚本不需要修改
参数设置分为两部分：

### 与系统相关的在main.py中的前部```getParameters()```函数中设置设置
- mode -- 设置train finetune test模式
- cuda -- 设置使用的GPU编号
- version -- 为本次实验的名称，此名称在每次不同的实验一定要进行修改，起名尽量具有实际意义
- experimentDescription -- 实验描述，描述本次实验的目的，以及记录一些必要的日志信息
- dataloader_workers -- dataloader的进程数，进程数越多数据载入越快但是CPU负载更大，一般不要设置太高，请设为6
- trainYaml -- 训练相关的配置文件，配置文件放置在./train_configs/中

### update.py使用
```update.py```可以更新本机此根目录下的文件到远程主机
修改```ssh_ip```、```ssh_username```、```ssh_passwd```、```ssh_port```来配置远程主机
```root_path```为远程主机的工程根目录
```scan_config```配置允许更新的文件后缀名

### 与训练相关的在./train_configs/*.yaml中
- trainScriptName -- 训练用的脚本名称```yourname```，放置在./train_scripts/中，命名方式为train_[yourname].py，训练用的脚本可以自行创建参考已有训练脚本文件
- gScriptName -- Generator的相关脚本文件名，放置在./components/中，可以自行创建修改该脚本文件需参考已有训练脚本文件
- dScriptName -- Discriminator的相关脚本文件名，放置在./components/中，可以自行创建修改该脚本文件需参考已有训练脚本文件
- selectedStyleDir -- 风格文件夹根目录中特定的风格文件目录
