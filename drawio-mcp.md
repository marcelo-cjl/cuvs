# Draw.io MCP 画图技能

## 概述

使用 next-ai-draw-io MCP Server 通过自然语言生成专业图表。

- **项目地址**: https://github.com/DayuanJiang/next-ai-draw-io
- **输出格式**: draw.io XML (.drawio)
- **视觉风格**: 专业/商务风格
- **特长**: 云架构图 (AWS/GCP/Azure 图标内置)

---

## 安装

```bash
claude mcp add drawio -- npx @next-ai-drawio/mcp-server@latest
```

验证安装：
```bash
claude mcp list
```

---

## 可用工具

| 工具 | 功能 |
|------|------|
| `start_session` | 打开浏览器实时预览 (端口 6002) |
| `create_new_diagram` | 从 XML 创建新图表 |
| `edit_diagram` | 通过 ID 编辑图表元素 |
| `get_diagram` | 获取当前图表的 XML |
| `export_diagram` | 导出为 .drawio 文件 |

---

## 使用流程

1. 先调用 `start_session` 打开浏览器预览
2. 用自然语言描述需要的图表
3. 调用 `create_new_diagram` 生成图表
4. 浏览器中实时查看效果
5. 如需修改，调用 `edit_diagram`
6. 完成后调用 `export_diagram` 导出文件

---

## 支持的图表类型

- 流程图 (Flowchart)
- 架构图 (Architecture Diagram)
- 云架构图 (AWS/GCP/Azure)
- 序列图 (Sequence Diagram)
- ER 图 (Entity Relationship)
- 网络拓扑图
- 组织架构图
- 思维导图

---

## 示例提示词

### 流程图
```
画一个用户登录流程图：
1. 用户输入账号密码
2. 验证账号是否存在
3. 验证密码是否正确
4. 成功则跳转首页，失败则提示错误
```

### AWS 架构图
```
画一个典型的 AWS 三层架构：
- 前端: CloudFront + S3
- 后端: ALB + ECS Fargate
- 数据库: RDS MySQL + ElastiCache Redis
```

### 系统架构图
```
画一个微服务架构图：
- API Gateway 接收请求
- 分发到 User Service, Order Service, Payment Service
- 各服务连接各自的数据库
- 通过 Kafka 进行服务间通信
```

---

## 备选方案: Excalidraw MCP

如果需要手绘风格或 Mermaid 支持：

```bash
claude mcp add excalidraw -- npx excalidraw-mcp
```

| 对比项 | draw.io MCP | Excalidraw MCP |
|--------|-------------|----------------|
| 风格 | 专业商务 | 手绘素描 |
| 云图标 | ✅ 内置 | ❌ 无 |
| Mermaid | ❌ | ✅ 支持 |
| 操作粒度 | 整图级 | 元素级 |

---

## 导出为 PNG

安装 draw.io 桌面版后可以命令行导出：

```bash
# 安装 draw.io
brew install --cask drawio

# 导出为 PNG
/opt/homebrew/bin/drawio --export --format png --output output.png input.drawio
```

---

## 完整工作流：画图 + 插入飞书文档

### 1. 画图并导出

```bash
# 使用 MCP 画图后导出
/opt/homebrew/bin/drawio --export --format png --output diagram.png diagram.drawio
```

### 2. 创建空图片 Block

```bash
curl -X POST "https://open.feishu.cn/open-apis/docx/v1/documents/$DOC_ID/blocks/$DOC_ID/children" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"children": [{"block_type": 27, "image": {}}]}'
```

### 3. 上传图片到 Block

```bash
curl -X POST "https://open.feishu.cn/open-apis/drive/v1/medias/upload_all" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -F "file_name=diagram.png" \
  -F "parent_type=docx_image" \
  -F "parent_node=$IMAGE_BLOCK_ID" \
  -F "size=$(stat -f%z diagram.png)" \
  -F "file=@diagram.png"
```

### 4. 更新 Block 显示图片

```bash
curl -X PATCH "https://open.feishu.cn/open-apis/docx/v1/documents/$DOC_ID/blocks/$IMAGE_BLOCK_ID" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"replace_image": {"token": "$FILE_TOKEN"}}'
```

### 所需权限

- `docx:document` - 创建/编辑文档
- `drive:drive` - 上传文件/图片

---

## 注意事项

1. 使用前需重启 Claude Code 加载 MCP
2. `start_session` 会在浏览器打开预览页面
3. 图表通过轮询方式实时更新
4. 导出的 .drawio 文件可用 draw.io 或 VS Code 插件打开编辑
5. 插入飞书文档需要 `drive:drive` 权限
