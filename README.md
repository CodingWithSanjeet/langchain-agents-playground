# 🦜🔗 LangChain v1.x & Agentic AI Master Study Notes

A comprehensive, beginner-friendly master reference guide covering **LangChain 1.x**, multi-provider LLM integrations (OpenAI, Google Gemini, Groq), custom tool definitions, state-graph AI agents built on **LangGraph**, real-time **streaming**, and parallel **batch processing**.

---

## 📋 Table of Contents
- [1. Quickstart & Project Setup](#1-quickstart--project-setup)
- [2. Module 1: AI Agents with LangChain v1.x (`create_agent`)](#2-module-1-ai-agents-with-langchain-v1x-create_agent)
  - [Core Concept: What is an Agent?](#core-concept-what-is-an-agent)
  - [Defining Custom Tools](#defining-custom-tools)
  - [Creating & Invoking the Agent](#creating--invoking-the-agent)
  - [The 4-Step Message Trajectory](#the-4-step-message-trajectory)
- [3. Module 2: Multi-Provider Model Integration](#3-module-2-multi-provider-model-integration)
  - [Initialization Methods: Factory vs. Direct Class](#initialization-methods-factory-vs-direct-class)
  - [Provider Integration Matrix](#provider-integration-matrix)
  - [Provider Code Implementations](#provider-code-implementations)
  - [Understanding `AIMessage` & Token Usage](#understanding-aimessage--token-usage)
- [4. Module 3: Real-Time Streaming & Parallel Batch Processing](#4-module-3-real-time-streaming--parallel-batch-processing)
  - [Real-Time Token Streaming (`model.stream`)](#real-time-token-streaming-modelstream)
  - [Parallel Batch Processing (`model.batch`)](#parallel-batch-processing-modelbatch)
  - [Concurrency Control (`max_concurrency`)](#concurrency-control-max_concurrency)
- [5. Revision Cheatsheet & Best Practices](#5-revision-cheatsheet--best-practices)

---

## 1. Quickstart & Project Setup

### Prerequisites
- Python `3.10+`
- Virtual environment tool (`uv` recommended)

### Environment & Installation
```bash
# 1. Create & activate virtual environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv add -r requirements.txt

# 3. Create .env file from template
cp .env.example .env
```

### Environment Variables (`.env`)
```env
OPENAI_API_KEY=sk-proj-...
GOOGLE_API_KEY=AQ.Ab8R...
GROQ_API_KEY=gsk_HYCe...
```

---

## 2. Module 1: AI Agents with LangChain v1.x (`create_agent`)

### Core Concept: What is an Agent?
An **AI Agent** uses a Large Language Model (LLM) as a dynamic reasoning engine. Instead of following a rigid, step-by-step code path, an agent inspects the user input, decides **if** an external tool is required, calls the tool with computed arguments, and synthesizes a final response.

> 💡 **LangChain 1.x Architecture**: Calling `create_agent()` compiles a state graph (`CompiledStateGraph`) powered by **LangGraph** under the hood.

---

### Defining Custom Tools
Tools are Python functions exposed to the LLM. 

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"the weather in {city} is sunny."
```

#### Why Docstrings & Type Annotations are Critical:
1. **Docstring (`"""Get the weather for a city."""`)**: The LLM reads this text to understand *what* the tool does and *when* to select it.
2. **Type Hint (`city: str`)**: The LLM uses this to construct arguments matching the expected schema.

---

### Creating & Invoking the Agent

```python
import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create Agent
agent = create_agent(
    model='gpt-5',                 # LLM Model identifier
    tools=[get_weather],           # List of callable tools
    system_prompt="You're a helpful assistant." # Global instructions
)

# Run Agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is the weather like in New York?"}]
})

# Extract Final Answer
final_text = response["messages"][-1].content
print(final_text)
```

---

### The 4-Step Message Trajectory

When an agent executes a tool call, the `response["messages"]` list records a 4-step communication loop:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. HumanMessage: "What is the weather in Bangalore?"                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. AIMessage: tool_calls=[{'name': 'get_weather', 'args': {'city': 'Bangalore'}}] │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. ToolMessage: "the weather in Bangalore is sunny."                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. AIMessage (Final Output): "It's sunny in Bangalore today."           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module 2: Multi-Provider Model Integration

LangChain provides a unified interface across different AI providers, letting you switch models seamlessly.

### Initialization Methods: Factory vs. Direct Class

1. **Unified Factory (`init_chat_model`)**:
   - **Syntax**: `init_chat_model("provider:model-name")`
   - **Advantage**: Provider-agnostic. Easily switch models via configuration string without editing python code.
2. **Direct Class (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatGroq`)**:
   - **Advantage**: Direct access to provider-specific configuration flags and custom endpoints.

---

### Provider Integration Matrix

| Provider | Target Model | Package Needed | Factory Pattern Syntax | Direct Class Import |
| :--- | :--- | :--- | :--- | :--- |
| **OpenAI** | `gpt-4.1` | `langchain-openai` | `init_chat_model("gpt-4.1")` | `ChatOpenAI(model="gpt-4.1")` |
| **Google Gemini** | `gemini-3.5-flash-lite` | `langchain-google-genai` | `init_chat_model("google_genai:gemini-3.5-flash-lite")` | `ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")` |
| **Groq** | `qwen/qwen3.8-27b` | `langchain-groq` | `init_chat_model("groq:qwen/qwen3.8-27b")` | `ChatGroq(model="qwen/qwen3.8-27b")` |

---

### Provider Code Implementations

#### 1. OpenAI Integration
```python
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI

# Method A: Factory Pattern
model_a = init_chat_model("gpt-4.1")

# Method B: Direct Class
model_b = ChatOpenAI(model="gpt-4.1")

res = model_a.invoke("Hello, how are you?")
print(res.content)
```

#### 2. Google Gemini Integration
```python
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI

# Method A: Factory Pattern (uses provider prefix)
model_a = init_chat_model("google_genai:gemini-3.5-flash-lite")

# Method B: Direct Class
model_b = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

res = model_a.invoke("Why do parrots talk?")
print(res.content)
```

#### 3. Groq Cloud Integration (Fast Open Weights)
```python
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq

# Method A: Factory Pattern (uses provider prefix)
model_a = init_chat_model("groq:qwen/qwen3.8-27b")

# Method B: Direct Class
model_b = ChatGroq(model="qwen/qwen3.8-27b")

res = model_a.invoke("Why do parrots talk?")
print(res.content)
```

---

### Understanding `AIMessage` & Token Usage

All model invocations return a standardized `AIMessage` object:

```python
response = model.invoke("Hello!")
```

```python
AIMessage(
    content="Hello! How can I help you today?",
    response_metadata={
        'model_name': 'gpt-4.1',
        'finish_reason': 'stop'
    },
    usage_metadata={
        'input_tokens': 10,
        'output_tokens': 12,
        'total_tokens': 22
    }
)
```

- **Extracting text**: `response.content`
- **Token cost tracking**: `response.usage_metadata['total_tokens']`

---

## 4. Module 3: Real-Time Streaming & Parallel Batch Processing

### Real-Time Token Streaming (`model.stream`)

Instead of waiting for the full response to generate, `.stream()` yields output chunks progressively as they arrive from the LLM provider.

```python
# Real-time streaming iterator
for chunk in model.stream("Write a short story about space."):
    print(chunk.text, end="", flush=True)
```

#### Key Benefits of Streaming:
- **Reduced Perceived Latency**: Users see immediate output within milliseconds instead of waiting 5–10 seconds.
- **ChatGPT-like Effect**: Perfect for user-facing chat UIs and terminal CLIs.

---

### Parallel Batch Processing (`model.batch`)

When you need to process multiple independent prompts, calling `.batch()` executes requests **in parallel** under the hood.

```python
prompts = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]

# Execute all prompts in parallel
responses = model.batch(prompts)

for response in responses:
    print(response.content)
```

#### Comparison: `.invoke()` vs `.batch()`
- **Sequential `.invoke()`**: Total time = $T_1 + T_2 + T_3 + \dots + T_n$
- **Parallel `.batch()`**: Total time $\approx \max(T_1, T_2, \dots, T_n)$

---

### Concurrency Control (`max_concurrency`)

To prevent exceeding LLM API rate limits (e.g. HTTP 429 Too Many Requests errors), configure maximum simultaneous parallel calls:

```python
responses = model.batch(
    prompts,
    config={
        "max_concurrency": 5  # Limit to 5 parallel workers
    }
)
```

---

## 5. Revision Cheatsheet & Best Practices

| Operation | Method / Best Practice | Description |
| :--- | :--- | :--- |
| **API Keys** | `dotenv.load_dotenv()` | Keep keys in `.env` and add `.env` to `.gitignore`. |
| **Single Invocation** | `model.invoke(prompt)` | Returns full response as `AIMessage`. |
| **Real-time Output** | `model.stream(prompt)` | Returns iterator of `AIMessageChunk` objects for live rendering. |
| **Parallel Tasks** | `model.batch([p1, p2, p3])` | Executes multiple independent prompts concurrently. |
| **Rate Limit Guard** | `config={"max_concurrency": N}` | Limits parallel API calls during `.batch()` execution. |
| **Agent Execution** | `agent.invoke({"messages": [...]})` | Executes state-graph tool-calling loop via LangGraph. |
| **Extracting Output** | `response["messages"][-1].content` | Gets final text response from an Agent run trajectory. |
