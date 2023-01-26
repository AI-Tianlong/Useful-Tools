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
