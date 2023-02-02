## Git相关的命令

### git clone 代理
```bash
git clone https://ghproxy.com/xxxxxxxxx
```
### git clone 一个仓库

```bash
git clone xxxxx
```

### 添加上游仓库

```bash
git remote add upstream xxxxxxx
```

## 添加pre-commit

```bash
pip install -U pre-commit
pre-commit install
pre-commit run --all-files
```

### 切换至dev 1.x分支

```bash
git checkout dev-1.x 
```

### 创建分支

```bash
git checkout -b AI-Tianlong/support_Mapillary_dataset
```

### git add+commit+push
```bash
git add .
git commit -m "xxxxx"       (--no-verify,有pre-commit hook的话)
git push                    (第一次 git push -u origin 分支名字)
```
### git 添加代理
```bash
# socks
git config --global http.proxy 'socks5://127.0.0.1:10808' 
git config --global https.proxy 'socks5://127.0.0.1:10808'
# http
git config --global http.proxy http://127.0.0.1:10809 
git config --global https.proxy https://127.0.0.1:10809

# 只对github.com使用代理，其他仓库不走代理
git config --global http.https://github.com.proxy socks5://127.0.0.1:10808
git config --global https.https://github.com.proxy socks5://127.0.0.1:10808
# 取消github代理
git config --global --unset http.https://github.com.proxy
git config --global --unset https.https://github.com.proxy

```