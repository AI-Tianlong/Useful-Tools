# Jetson AGX Orin 命令~
[[常见问题解答1]](https://forums.developer.nvidia.com/t/jetson-agx-orin-faq/237459)  
[[常见问题解答2]](https://forums.developer.nvidia.com/t/jetson-nano-faq/82953)
## 看内核版本
```bash
uname -r
```
## 命令行关闭电源
```bash
sudo poweroff
```

## jtop
[jtop的安装教程](https://blog.csdn.net/qq_48272604/article/details/122209295?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164706886216781683963530%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164706886216781683963530&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-122209295.pc_search_insert_es_download&utm_term=Jetson+nano%E6%B8%A9%E5%BA%A6%E5%AE%9E%E6%97%B6%E8%A7%82%E6%B5%8B&spm=1018.2226.3001.4187)
```bash
jtop
```
## 查看所用的jetson linux内核版本
```bash
cat /etc/nv_tegra_release
```
## 查看所用的jetson 型号
```bash
cat /etc/nv_boot_control.conf
```
## jtop显示不出来
```bash
export TERM='xterm-256color'
```

