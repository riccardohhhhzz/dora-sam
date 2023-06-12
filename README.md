# Cog-segment-anything

## 1	Get started

1. 生成ssh-key

```shell
ssh-keygen -t ed25519 -C "your_email@example.com"
```

1. 添加 ssh到个人github中

```shell
cat ~/.ssh/id_ed25519.pub
```

2. 进入到服务器

```shell
ssh ubuntu@<your-instance-ip>
```

3. 安装cog

```shell
sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
```

4. Clone  this project

```shell
git clone git@github.com:riccardohhhhzz/dora-sam.git
```

5. 下载模型文件

```shell
wget -O sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

6. 上传到replicate

```shell
sudo cog login
sudo cog push r8.im/<your-username>/<your-model-name>
```



## 2	Run example locally

```shell
sudo cog predict -i image=@imgs/88_0.jpg
```

