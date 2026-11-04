# legacy/

这里保留上一代 `bili_summary`（v2.5）的源码，仅作参考与迁移对照。

- `bili_summary_v2.py` — 旧的单文件实现（2272 行）：yt-dlp 下载整段音频 → faster-whisper 转写 → DeepSeek 总结
- `README_v2.md` — 旧版文档

新版 `aicbili` 已覆盖它的核心能力并增加了画面理解，差异见根目录 README 的「从 bili_summary 迁移」一节。
旧版里有而新版**未移植**的部分：`--setup` 设置向导、`register-cli` 注册全局命令、
以及针对超长音频的 whisper 分片/内存看门狗（新版改用云端 ASR 或 B站字幕，通常不再需要）。
