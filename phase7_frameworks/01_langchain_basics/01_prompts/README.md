# LangChain Prompts & Templates

**Purpose**: Learn LangChain's prompt template system and when to use it over f-strings

---

## Learning Objectives

By the end of this module, you will:
- Understand `PromptTemplate` for variable substitution
- Use `ChatPromptTemplate` for multi-message prompts
- Apply `FewShotPromptTemplate` for in-context learning
- Know when templates add value vs f-string simplicity
- Integrate templates with LangChain LLMs and chains

---

## Files

### `concepts.py` - Conceptual Understanding (No API Key Needed)
Run: `uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.concepts`

**Covers**:
- Template syntax and variable substitution
- Partial templates (pre-fill common variables)
- Chat message structure (system/user/assistant)
- Few-shot example formatting
- When to use templates vs f-strings

**No API required**: Shows template construction and formatting only

### `practical.py` - Hands-On with LangChain (Requires API Key)
Run: `uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.practical`

**Covers**:
- Templates with real LLM calls (ChatOpenAI)
- LCEL chain composition (template | llm)
- Output parsers for structured responses
- MessagesPlaceholder for chat history
- Real-world prompt engineering patterns

**Requires**: `OPENAI_API_KEY` in `.env`

---

## Key Concepts

### PromptTemplate
```python
from langchain.prompts import PromptTemplate

# basic template
template = PromptTemplate.from_template("Explain {topic} in simple terms")
prompt = template.format(topic="embeddings")

# partial templates (pre-fill some variables)
template = PromptTemplate(
    input_variables=["question"],
    partial_variables={"company": "Acme Corp"},
    template="You are a {company} support agent.\n\nUser: {question}"
)
```

### ChatPromptTemplate
```python
from langchain.prompts import ChatPromptTemplate

# multi-message prompt
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} assistant"),
    ("user", "Explain {topic}")
])

messages = template.format_messages(role="technical", topic="RAG")
```

### FewShotPromptTemplate
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"input": "happy", "output": "positive"},
    {"input": "sad", "output": "negative"}
]

example_template = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nSentiment: {output}"
)

few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    suffix="Input: {input}\nSentiment:",
    input_variables=["input"]
)
```

---

## When to Use What

### ✅ Use F-Strings (Phase 2) When:
- Simple, one-off prompts
- Few variables (<3)
- No reuse across codebase
- Maximum simplicity needed

### ✅ Use PromptTemplate When:
- Reusing prompts (DRY principle)
- Need input validation
- Partial template filling
- Combining with LangChain chains

### ✅ Use ChatPromptTemplate When:
- Multi-message prompts
- Injecting chat history
- Role-based prompting
- Building conversational agents

### ✅ Use FewShotPromptTemplate When:
- In-context learning with examples
- Many examples (>5)
- Dynamic example selection
- Token budget for examples

---

## Connection to Phase 2

**Phase 2 Approach**:
```python
# manual f-string formatting
topic = "embeddings"
prompt = f"Explain {topic} in simple terms"

# manual message construction
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": f"Explain {topic}"}
]
```

**LangChain Approach**:
```python
# reusable template with validation
template = PromptTemplate.from_template("Explain {topic} in simple terms")
prompt = template.format(topic="embeddings")

# structured chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Explain {topic}")
])
messages = chat_template.format_messages(topic="embeddings")
```

**Trade-off**: More abstraction vs more reusability and validation

---

## Next Steps

After completing this module:
1. Move to `02_llm_integration` (ChatOpenAI, ChatAnthropic)
2. Then `03_chains` (LCEL pipe operator)
3. Apply templates in real RAG/agent systems

---

## Resources

- [LangChain Prompt Templates Docs](https://python.langchain.com/docs/modules/model_io/prompts/)
- [Few-Shot Prompting Guide](https://python.langchain.com/docs/modules/model_io/prompts/few_shot_examples)
- [Chat Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/quick_start#chatprompttemplate)