# sam-infer

Interactive SAM segmentation server for the Anno annotation platform.

镜像会在首次启动时从 HuggingFace Hub 下载 SAM 权重（默认 `facebook/sam-vit-base`）。

## 快速开始

```bash
docker compose up --build
```

服务监听 `8422`（可用 `SERVER_PORT` 覆盖）。

## 配置

通过 `.env` 或环境变量配置，见 `.env.example`。常用项：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SAM_MODEL_ID` | `facebook/sam-vit-base` | HF 模型 id 或容器内本地权重路径 |
| `HF_CACHE_DIR` | `./hf-cache` | **宿主机**权重缓存目录（bind mount 到容器 `/app/.cache/huggingface`） |
| `SERVER_PORT` | `8422` | 对外端口 |
| `PROVIDER_API_KEY` | `changeme` | 接入鉴权 key |
| `DEVICE` | 自动检测 | 强制设备，如 `cpu` / `cuda` |

## 把模型权重存到外部 JuiceFS（或任意外部目录）

HF 下载的权重是大文件，可让它们落到外部存储（如 JuiceFS）上，便于迁移与跨机复用。
`HF_CACHE_DIR` 指向的宿主机目录会被 bind mount 到容器的 HF 缓存目录
（`/app/.cache/huggingface`，由 `HF_HOME` 决定），HF 下载会直接写到这里。

1. 确保宿主机已挂载 JuiceFS，例如 `/mnt/juicefs`，并建好目录：

   ```bash
   mkdir -p /mnt/juicefs/sam-infer/hf-cache
   ```

2. 在 `.env` 里指向该目录：

   ```bash
   HF_CACHE_DIR=/mnt/juicefs/sam-infer/hf-cache
   ```

3. 启动：

   ```bash
   docker compose up --build
   ```

   首次启动后，权重会以标准 HF cache 布局落到该目录，例如
   `/mnt/juicefs/sam-infer/hf-cache/hub/models--facebook--sam-vit-base/`。
   之后重启或换机，只要挂同一 `HF_CACHE_DIR`，即可直接复用缓存、无需重复下载。

> 容器以 root 运行，对已挂载的 JuiceFS 目录可正常读写。若目录权限受限，请确保容器对其可写。

### 使用预置的本地权重（离线可选）

若已把权重放在 `HF_CACHE_DIR` 对应目录下的某个子目录（标准 HF 目录布局），
可把 `SAM_MODEL_ID` 直接指向容器内路径，例如：

```bash
SAM_MODEL_ID=/app/.cache/huggingface/hub/models--facebook--sam-vit-base/snapshots/<hash>
```

未设置时按默认 hub id 联网下载，行为不变。
