---
tags: [主题簇, agent-dev模块, OpenGeoModeller]
aliases: [LLM路由, DeepSeek, CCR]
created: 2026-08-16
---

# 主题-LLM路由

同一套 Anthropic 协议客户端，后端可切 DeepSeek、Gemini、Grok、OpenRouter。配置互相覆盖，多文强调清理冲突的 `settings.json`。

## 节点

- [[Claude_Usage]] — DeepSeek 官方 API、本地网关 `:8787`、Gemini、claude-code-router `:3456`
- [[CodeX_Cursor]] — Cursor + DeepSeek MCP；Claude Desktop / CLI + Grok
- [[Skills]] — rightnow-cli / OpenRouter

## 交叉

Claude Code、Claude Desktop、Cursor 三套前端共享同一路由主题。

## 索引

- [[MOC-agent-dev]] · [[agent-dev模块关系]]
