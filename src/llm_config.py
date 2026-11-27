"""Runtime LLM configuration: compliance validation + connectivity check.

运行时 LLM 配置模块：把「地址、密钥、模型」集中成一个可校验的配置对象。

两道校验（前端配置面板与后端接口共用同一套规则，后端为准）：
1. 地址合规性（validate_base_url）：
   - 必须是 https；只有本机回环地址（localhost / 127.0.0.1 / ::1）允许 http；
   - 不允许携带用户名密码、查询串、锚点；
   - 主机必须在白名单内（主流 OpenAI 兼容端点），或为本机回环；
   - IP 字面量禁止私网 / 链路本地 / 保留 / 组播地址（防 SSRF）。
2. 连接校验（test_connection）：
   - 用最小 chat.completions 请求（max_tokens=1）验证密钥与端点真实可用，
     返回延迟、模型名与错误原因，不修改任何状态。

安全约束：配置只保存在进程内存中，不落盘、不写 .env、不回传完整密钥
（对外只返回掩码）。
"""

from __future__ import annotations

import ipaddress
import re
import time
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

#: 允许的 OpenAI 兼容端点主机（按需在此扩展）
ALLOWED_HOSTS: frozenset[str] = frozenset({
    "api.deepseek.com",
    "api.openai.com",
    "dashscope.aliyuncs.com",
    "open.bigmodel.cn",
    "api.moonshot.cn",
    "qianfan.baidubce.com",
})

#: 本机回环主机（允许 http，便于本地代理 / Ollama 等）
LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})

_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,64}$")
_KEY_RE = re.compile(r"^[\x21-\x7e]{8,512}$")


class LLMConfigError(ValueError):
    """配置不合规时抛出（前端可直接展示该消息）。"""


# ---------- 单字段校验 ----------

def validate_base_url(url: str) -> str:
    """校验并规范化 API 地址，返回去掉末尾斜杠的地址。"""
    raw = (url or "").strip()
    if not raw:
        raise LLMConfigError("API 地址不能为空")
    if "://" not in raw:
        raise LLMConfigError("API 地址必须以 https:// 开头")

    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()

    if parts.username or parts.password:
        raise LLMConfigError("API 地址不允许携带用户名或密码")
    if parts.query or parts.fragment:
        raise LLMConfigError("API 地址不允许包含查询串或锚点")
    if not host:
        raise LLMConfigError("API 地址缺少主机名")

    is_loopback = host in LOOPBACK_HOSTS
    if scheme == "http" and not is_loopback:
        raise LLMConfigError("非本机地址必须使用 https")
    if scheme not in ("https", "http"):
        raise LLMConfigError("仅支持 https（本机可用 http）")

    # IP 字面量：只允许回环，杜绝内网探测
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None:
        if not (ip.is_loopback):
            raise LLMConfigError("不允许使用 IP 地址作为端点（本机回环除外）")
    elif not is_loopback and host not in ALLOWED_HOSTS:
        allowed = "、".join(sorted(ALLOWED_HOSTS))
        raise LLMConfigError(f"端点不在白名单内，可用：{allowed}（本机回环地址除外）")

    path = parts.path.rstrip("/")
    netloc = parts.netloc
    return f"{scheme}://{netloc}{path}"


def validate_model(model: str) -> str:
    """校验模型名（字母数字与 . _ : -，1~64 字符）。"""
    name = (model or "").strip()
    if not name:
        raise LLMConfigError("模型名不能为空")
    if not _MODEL_RE.match(name):
        raise LLMConfigError("模型名只能包含字母、数字与 . _ : -（1~64 字符）")
    return name


def validate_api_key(key: str) -> str:
    """校验密钥格式（可打印 ASCII，8~512 字符），不做联网验证。"""
    k = (key or "").strip()
    if not k:
        raise LLMConfigError("API Key 不能为空")
    if not _KEY_RE.match(k):
        raise LLMConfigError("API Key 格式不合法（应为 8~512 位可打印字符）")
    return k


# ---------- 配置对象 ----------

@dataclass(frozen=True)
class LLMConfig:
    """一份可用的 LLM 配置（构造时即校验）。"""

    base_url: str
    api_key: str
    model: str = "deepseek-chat"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", validate_base_url(self.base_url))
        object.__setattr__(self, "api_key", validate_api_key(self.api_key))
        object.__setattr__(self, "model", validate_model(self.model))
        t = float(self.temperature)
        if not (0.0 <= t <= 2.0):
            raise LLMConfigError("temperature 必须在 0~2 之间")
        object.__setattr__(self, "temperature", t)

    # 对外展示：绝不返回完整密钥
    def masked(self) -> dict:
        key = self.api_key
        tail = key[-4:] if len(key) >= 8 else ""
        return {
            "base_url": self.base_url,
            "host": urlsplit(self.base_url).hostname,
            "model": self.model,
            "temperature": self.temperature,
            "api_key_masked": f"****{tail}",
        }

    def with_temperature(self, temperature: float) -> "LLMConfig":
        return replace(self, temperature=temperature)

    @classmethod
    def from_payload(cls, payload: dict) -> "LLMConfig":
        """从请求体构造配置（字段缺失时给出明确报错）。"""
        return cls(
            base_url=str(payload.get("base_url", "")),
            api_key=str(payload.get("api_key", "")),
            model=str(payload.get("model", "deepseek-chat")),
            temperature=float(payload.get("temperature", 0.0) or 0.0),
        )


# ---------- 连接校验 ----------

def test_connection(cfg: LLMConfig, timeout: float = 20.0) -> dict:
    """发一个最小请求验证端点与密钥可用；返回结构化结果，不抛异常。"""
    started = time.perf_counter()
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - 依赖缺失时给出可读提示
        return {"ok": False, "latency_ms": None, "model": cfg.model,
                "message": f"缺少 openai 依赖：{e}"}

    try:
        client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=timeout,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        latency = round((time.perf_counter() - started) * 1000)
        return {
            "ok": True,
            "latency_ms": latency,
            "model": getattr(resp, "model", cfg.model) or cfg.model,
            "message": "连接成功",
        }
    except Exception as e:  # noqa: BLE001 - 需要把任意网络/鉴权错误转成可读文本
        latency = round((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "latency_ms": latency,
            "model": cfg.model,
            "message": f"{type(e).__name__}: {e}",
        }
