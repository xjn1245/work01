# server 目录说明

该目录包含服务端入口相关逻辑：

- `routes.py`：主要 API 路由定义（聊天、头像、管理后台、鉴权等）。
- `state.py`：`AppState` 数据结构，维护运行时共享状态。
- `auth.py` / `auth_store.py`：认证与账号存储。
- `avatar_admin_store.py`：数字人后台配置（SQLite）落库实现。
- `chat_history.py`：会话历史落库与查询。

