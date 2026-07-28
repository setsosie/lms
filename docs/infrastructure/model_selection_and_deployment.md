> ⚠️ **SUPERSEDED (2026-06-18).** This Dec-2025 plan (DeepSeek-V2 + Qwen2.5-Math-72B menu)
> is **stale**. The binding logic has shifted from "one big MLA model for 128K context" to
> "cheap-to-serve low-active MoEs run in many oracle-checked attempts." Use
> **`docs/infrastructure/2026Q2-lean-codegen-base-model-selection.md`** instead. Kept for
> history only.

# Model Selection and Deployment for LMS

**Status**: SUPERSEDED — see 2026Q2-lean-codegen-base-model-selection.md
**Date**: 2025-12-24
**Hardware**: 4× H100 NVL (376GB VRAM total)
**Goal**: 50 heterogeneous agents for mathematical formalization

---

## 1. Executive Summary

This document captures research on deploying open-source MoE (Mixture-of-Experts) and dense models for the LMS project. Key findings:

1. **DeepSeek-V2** with MLA attention reduces KV cache by 93.3%, enabling 128K context per agent
2. **Heterogeneous collectives** align with Henrich's collective brain theory - cognitive diversity may outperform homogeneous scaling
3. **vLLM** has first-class MoE support but requires separate instances per model
4. **50 agents** are feasible with careful memory partitioning

---

## 2. Lessons from AI Engineering (Chip Huyen)

### 2.1 Agent Architecture Validation

The book validates our core approach:

| AI Engineering Concept | LMS Application |
|------------------------|-----------------|
| Multi-agent with verification | LEAN as perfect verification oracle |
| Compound error (95%^10 = 60%) | LEAN resets errors at each verified step |
| Reflection patterns (ReAct) | Working groups critique proofs |
| Tool evolution / skill manager | Proven lemmas become reusable tools |
| Data flywheel | Verified proofs become training signal |
| Specialized judges | LEAN is the ultimate specialized judge |

### 2.2 Key Insight: Test-Time Compute

> "Using a verifier resulted in approximately the same performance boost as a 30× model size increase."

This validates LMS fundamentally: LEAN verification + multiple attempts > larger model alone.

### 2.3 Warning: Model Collapse

When agents train on each other's outputs:
- Probable events become over-represented
- Rare proof strategies get forgotten
- Mitigate by mixing human-written proofs in training

---

## 3. Hardware: 4× H100 NVL

| Spec | Value |
|------|-------|
| GPU Memory | 4 × 94GB = 376GB |
| Interconnect | NVLink |
| Memory Bandwidth | 3.35 TB/s per GPU |

---

## 4. Model Menu

### 4.1 Large MoE (Strategic Reasoning)

| Model | Total Params | Active | VRAM (Quant) | Context | Notes |
|-------|--------------|--------|--------------|---------|-------|
| **DeepSeek-V2** | 236B | 21B | ~136GB (4-bit) | 128K | MLA = 93% KV cache reduction |
| **Mixtral 8x22B** | 141B | 39B | ~150GB (FP8) | 64K | Strong math/code |
| **Mixtral 8x7B** | 46.7B | 12.9B | ~25GB (4-bit) | 32K | Fast, fits easily |
| **Qwen2-MoE** | 57B | 14B | ~60GB (FP8) | 32K | Good multilingual |

### 4.2 Dense Math Specialists

| Model | Params | VRAM (Quant) | MATH Score | Notes |
|-------|--------|--------------|------------|-------|
| **Qwen2.5-Math-72B** | 72B | ~80GB (4-bit) | 83.6% | Best open math model |
| **DeepSeek-R1-Distill-32B** | 32B | ~36GB (4-bit) | 72.6% AIME | Explicit reasoning |
| **Qwen2.5-Math-7B** | 7B | ~8GB (4-bit) | 83.6% (TIR) | Fast, strong for size |
| **DeepSeek-R1-Distill-7B** | 7B | ~8GB (4-bit) | Good | R1 reasoning distilled |

### 4.3 Why DeepSeek-V2 for LMS?

DeepSeek's Multi-head Latent Attention (MLA) compresses KV cache dramatically:

```
Standard Attention: KV cache = 2 × layers × kv_heads × head_dim × bytes
MLA: KV cache = 2 × layers × latent_dim × bytes (latent_dim << kv_heads × head_dim)

Result: 93.3% reduction in KV cache memory
```

For 50 agents:
- **Standard (Mixtral)**: ~20K context max per agent
- **MLA (DeepSeek-V2)**: Full 128K context per agent

---

## 5. 50-Agent Architecture

### 5.1 Agent Distribution

| Role | Count | Purpose |
|------|-------|---------|
| **Planning Panel** | 5 | Strategic task allocation |
| **Working Groups** | 40 | 10 groups × 4 agents each |
| **Review Panel** | 5 | Harsh verification review |
| **Total** | **50** | |

### 5.2 Memory Budget (Heterogeneous)

| Role | Model | Count | VRAM |
|------|-------|-------|------|
| Planners + Reviewers | DeepSeek-V2 | 10 | 136GB (shared) |
| Math Workers | Qwen2.5-Math-72B | 15 | 80GB |
| Reasoning Workers | DeepSeek-R1-Distill-32B | 15 | 36GB |
| Fast Explorers | Mixtral 8x7B | 10 | 25GB |
| **Total** | | **50** | **~277GB** |

Leaves ~100GB for KV cache overhead.

---

## 6. Deployment Options

### Option A: Sequential Model Swapping

```
Phase 1: DeepSeek-V2 (Planning) → 136GB, 128K context
Phase 2: Mixtral 8x7B (Working) → 25GB, fast throughput
Phase 3: DeepSeek-V2 (Review)  → 136GB, deep analysis
```

**Pros:** Full memory per model, maximum context
**Cons:** ~30-60s swap latency between phases

### Option B: Partitioned GPUs (Recommended for 50 agents)

```
┌─────────────────────────────────────────────────────────┐
│  GPU 0-1 (188GB)              │  GPU 2-3 (188GB)        │
│  ─────────────────            │  ─────────────────      │
│  DeepSeek-V2 (4-bit)          │  Mixed small models:    │
│  136GB + KV cache             │  • Qwen2.5-Math-72B     │
│                               │  • Mixtral 8x7B         │
│  Planning + Review agents     │                         │
│  (10 agents @ 128K context)   │  Working group agents   │
│                               │  (40 agents @ 16-32K)   │
└─────────────────────────────────────────────────────────┘
```

### Option C: Tiered Architecture (Maximum Diversity)

```
┌────────────────────────────────────────────────────────────┐
│                    4×H100 NVL (376GB)                      │
├────────────────────────────────────────────────────────────┤
│  "Oracle" Layer (GPU 0-1): DeepSeek-V2 136GB               │
│  └─ 5 Planning agents (128K context)                       │
│  └─ 5 Review agents (128K context)                         │
├────────────────────────────────────────────────────────────┤
│  "Specialist" Layer (GPU 2): Qwen2.5-Math-72B 80GB         │
│  └─ 10 Math-focused agents (32K context)                   │
├────────────────────────────────────────────────────────────┤
│  "Worker" Layer (GPU 3): Multiple small models ~90GB       │
│  └─ Mixtral 8x7B (25GB) - 15 general workers               │
│  └─ DeepSeek-R1-Distill-7B (8GB) - 10 reasoning workers    │
│  └─ Qwen2.5-Math-7B (8GB) - 5 fast math workers            │
└────────────────────────────────────────────────────────────┘
```

---

## 7. Python Implementation

### 7.1 Provider Extension for vLLM

```python
# lms/providers/vllm_local.py

from dataclasses import dataclass
from openai import AsyncOpenAI
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.config import ProviderConfig


@dataclass
class VLLMEndpoint:
    """A vLLM server endpoint."""

    name: str
    base_url: str  # e.g., "http://localhost:8000/v1"
    model: str     # e.g., "deepseek-ai/DeepSeek-V2"
    max_context: int = 128000
    gpu_ids: list[int] = None  # Which GPUs this instance uses


class VLLMProvider(BaseLLMProvider):
    """Provider for local vLLM instances."""

    name = "vllm"

    def __init__(self, config: ProviderConfig, endpoints: list[VLLMEndpoint]):
        super().__init__(config)
        self.endpoints = {e.name: e for e in endpoints}
        self.clients = {
            name: AsyncOpenAI(base_url=e.base_url, api_key="not-needed")
            for name, e in self.endpoints.items()
        }

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        endpoint_name: str | None = None,  # Which vLLM instance to use
    ) -> GenerationResponse:
        """Generate using a specific vLLM endpoint."""

        endpoint_name = endpoint_name or list(self.endpoints.keys())[0]
        endpoint = self.endpoints[endpoint_name]
        client = self.clients[endpoint_name]

        # Build messages
        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        response = await client.chat.completions.create(
            model=endpoint.model,
            messages=api_messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
        )

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        self._track_usage(usage)

        return GenerationResponse(
            content=response.choices[0].message.content,
            usage=usage,
            provider=f"vllm:{endpoint_name}",
        )
```

### 7.2 Heterogeneous Model Router

```python
# lms/providers/router.py

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message
from lms.providers.vllm_local import VLLMProvider, VLLMEndpoint


class AgentTier(Enum):
    """Agent tiers determine which model pool to use."""

    ORACLE = "oracle"       # Planning, Review (DeepSeek-V2)
    SPECIALIST = "specialist"  # Math specialists (Qwen2.5-Math-72B)
    WORKER = "worker"       # Fast workers (Mixtral 8x7B, small models)


@dataclass
class ModelPool:
    """A pool of models for a specific tier."""

    tier: AgentTier
    endpoints: list[str]  # Endpoint names in VLLMProvider
    context_limit: int
    description: str


class HeterogeneousRouter:
    """Routes agent requests to appropriate model pools."""

    def __init__(self, provider: VLLMProvider, pools: list[ModelPool]):
        self.provider = provider
        self.pools = {p.tier: p for p in pools}
        self._request_counts = {tier: 0 for tier in AgentTier}

    def get_endpoint_for_agent(self, agent_id: str, tier: AgentTier) -> str:
        """Get the endpoint name for an agent based on tier."""

        pool = self.pools[tier]
        # Round-robin within pool
        idx = self._request_counts[tier] % len(pool.endpoints)
        self._request_counts[tier] += 1
        return pool.endpoints[idx]

    async def generate(
        self,
        messages: list[Message],
        agent_id: str,
        tier: AgentTier,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResponse:
        """Generate using the appropriate model for agent tier."""

        endpoint = self.get_endpoint_for_agent(agent_id, tier)
        return await self.provider.generate(
            messages=messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            endpoint_name=endpoint,
        )


# Example configuration for Option B (Partitioned GPUs)
def create_option_b_router() -> HeterogeneousRouter:
    """Create router for Option B: Partitioned GPUs."""

    endpoints = [
        VLLMEndpoint(
            name="deepseek-v2",
            base_url="http://localhost:8000/v1",
            model="deepseek-ai/DeepSeek-V2",
            max_context=128000,
            gpu_ids=[0, 1],
        ),
        VLLMEndpoint(
            name="qwen-math-72b",
            base_url="http://localhost:8001/v1",
            model="Qwen/Qwen2.5-Math-72B-Instruct",
            max_context=32000,
            gpu_ids=[2],
        ),
        VLLMEndpoint(
            name="mixtral-8x7b",
            base_url="http://localhost:8002/v1",
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_context=32000,
            gpu_ids=[3],
        ),
    ]

    provider = VLLMProvider(
        config=ProviderConfig(provider="vllm", model="multi"),
        endpoints=endpoints,
    )

    pools = [
        ModelPool(
            tier=AgentTier.ORACLE,
            endpoints=["deepseek-v2"],
            context_limit=128000,
            description="Planning and Review (DeepSeek-V2 128K)",
        ),
        ModelPool(
            tier=AgentTier.SPECIALIST,
            endpoints=["qwen-math-72b"],
            context_limit=32000,
            description="Math specialists (Qwen2.5-Math-72B)",
        ),
        ModelPool(
            tier=AgentTier.WORKER,
            endpoints=["mixtral-8x7b"],
            context_limit=32000,
            description="Fast workers (Mixtral 8x7B)",
        ),
    ]

    return HeterogeneousRouter(provider, pools)
```

### 7.3 Option A: Sequential Swapping

```python
# lms/providers/swapping.py

import asyncio
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelSpec:
    """Specification for a model to be loaded."""

    name: str
    model_id: str
    quantization: str | None = None  # "awq", "gptq", "fp8", None
    tensor_parallel: int = 4
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 128000


class ModelSwapper:
    """Manages swapping models in and out of GPU memory."""

    def __init__(self, vllm_port: int = 8000):
        self.port = vllm_port
        self.current_model: str | None = None
        self.process: subprocess.Popen | None = None

    async def load_model(self, spec: ModelSpec, wait_ready: bool = True) -> None:
        """Load a model, stopping any currently running model."""

        if self.current_model == spec.name:
            return  # Already loaded

        # Stop current model
        await self.unload_model()

        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.model_id,
            "--port", str(self.port),
            "--tensor-parallel-size", str(spec.tensor_parallel),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            "--max-model-len", str(spec.max_model_len),
        ]

        if spec.quantization:
            cmd.extend(["--quantization", spec.quantization])

        # Start vLLM server
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.current_model = spec.name

        if wait_ready:
            await self._wait_for_ready()

    async def unload_model(self) -> None:
        """Stop the current model server."""

        if self.process:
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None
            self.current_model = None

            # Wait for GPU memory to be released
            await asyncio.sleep(5)

    async def _wait_for_ready(self, timeout: int = 120) -> None:
        """Wait for vLLM server to be ready."""

        import aiohttp

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:{self.port}/health"
                    ) as resp:
                        if resp.status == 200:
                            return
            except:
                pass
            await asyncio.sleep(2)

        raise TimeoutError(f"vLLM server not ready after {timeout}s")


class SequentialSwappingProvider:
    """Provider that swaps models between phases."""

    def __init__(self):
        self.swapper = ModelSwapper()

        self.models = {
            "planning": ModelSpec(
                name="planning",
                model_id="deepseek-ai/DeepSeek-V2",
                quantization="awq",
                max_model_len=128000,
            ),
            "working": ModelSpec(
                name="working",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                quantization=None,
                max_model_len=32000,
            ),
            "review": ModelSpec(
                name="review",
                model_id="deepseek-ai/DeepSeek-V2",
                quantization="awq",
                max_model_len=128000,
            ),
        }

    async def run_phase(self, phase: str, work_fn) -> any:
        """Run a phase with the appropriate model loaded."""

        spec = self.models[phase]
        await self.swapper.load_model(spec)

        # Run the work
        result = await work_fn()

        return result
```

### 7.4 Option C: Tiered with LiteLLM Proxy

```python
# lms/providers/tiered.py

"""
Option C uses LiteLLM as a unified proxy to multiple vLLM instances.

Start vLLM instances:
  GPU 0-1: vllm serve deepseek-ai/DeepSeek-V2 --port 8000 --tp 2
  GPU 2:   vllm serve Qwen/Qwen2.5-Math-72B-Instruct --port 8001
  GPU 3:   vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --port 8002
           vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --port 8003

Then configure LiteLLM (config.yaml):
  model_list:
    - model_name: oracle
      litellm_params:
        model: openai/deepseek-ai/DeepSeek-V2
        api_base: http://localhost:8000/v1
    - model_name: math-specialist
      litellm_params:
        model: openai/Qwen/Qwen2.5-Math-72B-Instruct
        api_base: http://localhost:8001/v1
    - model_name: worker-general
      litellm_params:
        model: openai/mistralai/Mixtral-8x7B-Instruct-v0.1
        api_base: http://localhost:8002/v1
    - model_name: worker-reasoning
      litellm_params:
        model: openai/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
        api_base: http://localhost:8003/v1

Start LiteLLM:
  litellm --config config.yaml --port 4000
"""

from dataclasses import dataclass
from openai import AsyncOpenAI
from lms.providers.base import BaseLLMProvider, GenerationResponse, Message, TokenUsage
from lms.config import ProviderConfig


@dataclass
class TieredAgentConfig:
    """Configuration for an agent in the tiered system."""

    agent_id: str
    tier: str  # "oracle", "math-specialist", "worker-general", "worker-reasoning"
    role: str  # "planner", "reviewer", "math-worker", "explorer", etc.
    context_budget: int


class TieredProvider(BaseLLMProvider):
    """Provider using LiteLLM proxy for tiered model access."""

    name = "tiered"

    # Map agent roles to model tiers
    ROLE_TO_TIER = {
        # Planning Panel
        "planning-chair": "oracle",
        "planning-member": "oracle",
        # Review Panel
        "review-chair": "oracle",
        "review-member": "oracle",
        # Working Groups
        "wg-chair": "worker-general",
        "wg-scribe": "worker-general",
        "wg-researcher-math": "math-specialist",
        "wg-researcher-general": "worker-general",
        "wg-researcher-reasoning": "worker-reasoning",
    }

    def __init__(self, config: ProviderConfig, litellm_base_url: str = "http://localhost:4000"):
        super().__init__(config)
        self.client = AsyncOpenAI(base_url=litellm_base_url, api_key="not-needed")

    def get_model_for_role(self, role: str) -> str:
        """Get the LiteLLM model name for an agent role."""
        return self.ROLE_TO_TIER.get(role, "worker-general")

    async def generate(
        self,
        messages: list[Message],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        role: str = "worker-general",
    ) -> GenerationResponse:
        """Generate using the appropriate tiered model."""

        model = self.get_model_for_role(role)

        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        response = await self.client.chat.completions.create(
            model=model,
            messages=api_messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
        )

        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        self._track_usage(usage)

        return GenerationResponse(
            content=response.choices[0].message.content,
            usage=usage,
            provider=f"tiered:{model}",
        )
```

### 7.5 Integration with Working Groups

```python
# lms/working_group.py (additions)

from enum import Enum
from lms.providers.router import AgentTier, HeterogeneousRouter


class WorkerSpecialization(Enum):
    """Specializations for working group researchers."""

    MATH = "math"           # Use math-specialist model
    REASONING = "reasoning" # Use reasoning model
    GENERAL = "general"     # Use general worker model


def assign_specializations(group_id: int, n_researchers: int) -> list[WorkerSpecialization]:
    """Assign specializations to researchers in a group.

    Strategy: Mix specializations for cognitive diversity.
    """

    if n_researchers == 1:
        return [WorkerSpecialization.MATH]
    elif n_researchers == 2:
        return [WorkerSpecialization.MATH, WorkerSpecialization.REASONING]
    else:
        # For 3+, ensure at least one of each
        specs = [
            WorkerSpecialization.MATH,
            WorkerSpecialization.REASONING,
            WorkerSpecialization.GENERAL,
        ]
        # Fill remaining with round-robin
        while len(specs) < n_researchers:
            specs.append(specs[len(specs) % 3])
        return specs[:n_researchers]


class HeterogeneousWorkingGroup(WorkingGroup):
    """Working group that uses different models for different roles."""

    def __init__(
        self,
        config: WorkingGroupConfig,
        router: HeterogeneousRouter,
        foundation_summary: str,
    ):
        self.config = config
        self.router = router
        self.foundation_summary = foundation_summary
        self.state = WorkingGroupState(config=config)

        # Assign specializations
        n_researchers = config.members_per_role.get(Role.RESEARCHER, 1)
        self.specializations = assign_specializations(config.group_id, n_researchers)

        # Create member IDs with specializations
        self.members: list[tuple[str, Role, AgentTier]] = []

        # Chair uses WORKER tier (doesn't need deep reasoning)
        self.members.append((
            f"group-{config.group_id}-chair",
            Role.CHAIR,
            AgentTier.WORKER,
        ))

        # Scribe uses WORKER tier
        self.members.append((
            f"group-{config.group_id}-scribe",
            Role.SCRIBE,
            AgentTier.WORKER,
        ))

        # Researchers get varied tiers based on specialization
        for i, spec in enumerate(self.specializations):
            if spec == WorkerSpecialization.MATH:
                tier = AgentTier.SPECIALIST
            else:
                tier = AgentTier.WORKER

            self.members.append((
                f"group-{config.group_id}-researcher-{spec.value}-{i}",
                Role.RESEARCHER,
                tier,
            ))

    async def _generate_for_member(
        self,
        member_id: str,
        tier: AgentTier,
        messages: list,
        system_prompt: str,
    ) -> str:
        """Generate response using the appropriate model for this member."""

        response = await self.router.generate(
            messages=messages,
            agent_id=member_id,
            tier=tier,
            system_prompt=system_prompt,
        )
        return response.content
```

---

## 8. Startup Scripts

### 8.1 Option B: Partitioned GPUs

```bash
#!/bin/bash
# scripts/start_option_b.sh

# GPU 0-1: DeepSeek-V2 (Planning + Review)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 128000 \
    --quantization awq &

# GPU 2: Qwen2.5-Math-72B (Math specialists)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-72B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32000 \
    --quantization gptq &

# GPU 3: Mixtral 8x7B (Fast workers)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --port 8002 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32000 &

wait
```

### 8.2 Option C: Full Tiered Setup

```bash
#!/bin/bash
# scripts/start_option_c.sh

# GPU 0-1: Oracle (DeepSeek-V2)
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-V2 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 128000 \
    --quantization awq &

# GPU 2: Math specialist (Qwen2.5-Math-72B)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-72B-Instruct \
    --port 8001 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32000 \
    --quantization gptq &

# GPU 3: Split between small models
# Mixtral 8x7B (25GB)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --port 8002 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 16000 &

# DeepSeek-R1-Distill-7B (8GB)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --port 8003 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 16000 &

# Qwen2.5-Math-7B (8GB)
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-7B-Instruct \
    --port 8004 \
    --gpu-memory-utilization 0.15 \
    --max-model-len 16000 &

# Start LiteLLM proxy
litellm --config litellm_config.yaml --port 4000 &

wait
```

---

## 9. Throughput Estimates

### 9.1 Per-Model Throughput (50 concurrent agents)

| Model | Total tok/s | Per Agent | Turn Time (500 tok) |
|-------|-------------|-----------|---------------------|
| DeepSeek-V2 (10 agents) | ~400-600 | ~40-60 tok/s | ~10 sec |
| Qwen2.5-Math-72B (15 agents) | ~300-450 | ~20-30 tok/s | ~20 sec |
| Mixtral 8x7B (25 agents) | ~800-1200 | ~32-48 tok/s | ~12 sec |

### 9.2 Full Generation Cycle

```
Planning Phase:   5 agents × 3 turns × 10 sec = ~2.5 min
Working Phase:   40 agents × 5 turns × 15 sec = ~12.5 min (parallel groups)
Review Phase:     5 agents × 3 turns × 10 sec = ~2.5 min
─────────────────────────────────────────────────────────
Total per generation: ~17-20 minutes
```

---

## 10. Future: 8×H100 NVL / 8×H200 Dream Setup

With 8×H100 NVL (752GB) or 8×H200 (1.1TB):

| Model | Precision | VRAM | Agents | Context |
|-------|-----------|------|--------|---------|
| **DeepSeek-V3** | 4-bit | 400GB | 20 | 128K |
| **DeepSeek-V2** | FP8 | 280GB | 30 | 128K |
| **Qwen2.5-Math-72B** | FP16 | 150GB | 40 | 64K |
| **Specialists pool** | 4-bit | 100GB | 60 | 32K |

Could run **150+ agents** with full context and top-tier models.

---

## 11. References

- [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [DeepSeek-V2 MLA Paper](https://arxiv.org/abs/2405.04434)
- [DeepSeek MLA Explanation](https://medium.com/foundation-models-deep-dive/deepseeks-multi-head-latent-attention-mla-is-shrinking-the-kv-cache-27328f7dda27)
- [Qwen2.5-Math](https://qwenlm.github.io/blog/qwen2.5-math/)
- [vLLM Multi-Model Discussion](https://github.com/vllm-project/vllm/issues/3326)
- [AI Engineering (Chip Huyen, 2024)](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)
