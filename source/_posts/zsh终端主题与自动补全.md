---
title: Zsh 环境配置指南
date: 2025-10-29 19:25:27
categories: 
  - "linux"
tags:
  - linux

---

# ⚙️  Zsh 环境配置指南

本指南介绍如何在系统上配置现代化终端环境，包含：

- 安装 **Zsh**

- 安装 **Oh My Zsh**

- 安装主题 **Powerlevel10k**

- 安装 **命令自动补全** 与 **语法高亮**

  效果如下：

  ![zsh](/images/zsh/zsh.png)

  本教程使用国内源

---

## 📦 1. 安装 Zsh并设置

更新系统并安装 Zsh：

```bash
sudo apt update
sudo apt install -y zsh git curl
```

验证是否安装成功

```shell
zsh --version
# zsh 5.9 (aarch64-unknown-linux-gnu)
```

设置 Zsh 为默认 Shell，并注销重启终端

```shell
chsh -s $(which zsh)
```

## 📦 2. 安装 Oh My Zsh

Oh My Zsh 是基于 zsh 命令行的一个扩展工具集，提供了丰富的扩展功能。

通过 curl 安装：

```SHELL
export REMOTE=https://gitee.com/mirrors/oh-my-zsh.git /
export ZSH="$HOME/.oh-my-zsh" /
git clone --depth=1 $REMOTE $ZSH /
cp $ZSH/templates/zshrc.zsh-template ~/.zshrc
```

## 📦 3. 安装 PowerLevel10K主题与插件

安装p10k主题

```Shell
git clone --depth=1 https://gitee.com/wulkha/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k
```

安装命令自动补全插件 (zsh-autosuggestions)

```shell
git clone https://gitee.com/wulkha/zsh-autosuggestions.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

安装语法高亮插件 (zsh-syntax-highlighting)

```SHELL
git clone https://gitee.com/wulkha/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

所有下载完成后，编辑~/.zshrc，修改以下两个

```shell
ZSH_THEME="powerlevel10k/powerlevel10k"
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
```

激活环境

```shell
source ~/.zshrc
```

## 📦 4. 补充

powerlevel10k主题会在第一次启动配置，若后期想更改，可以执行以下命令

```shell
p10k configure
```

若遇到显示不全，需要下载Nerd Fonts字体，远程连接，需要配置终端字体，下载即可

[点击下载 MesloLGS NF Regular.ttf](/images/zsh/MesloLGS%20NF%20Regular.ttf)