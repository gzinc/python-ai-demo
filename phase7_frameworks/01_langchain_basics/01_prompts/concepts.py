"""
LangChain Prompts - Conceptual Understanding

This module demonstrates LangChain's prompt template system without requiring an API key.
Shows template construction, formatting, and when to use each type vs Phase 2 f-strings.

Run: uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.concepts

No API key required - demonstrates patterns through output inspection.
"""

from inspect import cleandoc
from typing import Any

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate


# region Helper Functions


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_subsection(title: str) -> None:
    """print subsection header"""
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print('-' * 70)


def print_output(label: str, value: Any) -> None:
    """print labeled output"""
    print(f"\n{label}:")
    print(f"  {value}")


# endregion


# region 1. Basic PromptTemplate vs F-Strings


def demo_basic_templates() -> None:
    """demonstrate basic PromptTemplate vs f-strings"""
    print_section("1. Basic PromptTemplate vs F-Strings")

    # Phase 2 approach (f-strings)
    print_subsection("Phase 2 Approach: F-Strings")
    topic = "embeddings"
    phase2_prompt = f"Explain {topic} in simple terms"
    print_output("F-String Result", phase2_prompt)

    # LangChain approach (PromptTemplate)
    print_subsection("LangChain Approach: PromptTemplate")
    template = PromptTemplate.from_template("Explain {topic} in simple terms")
    langchain_prompt = template.format(topic="embeddings")
    print_output("PromptTemplate Result", langchain_prompt)

    print_subsection("Why Use PromptTemplate?")
    benefits = cleandoc('''
        ✅ Reusability: Define once, use many times
        ✅ Validation: Ensures all required variables are provided
        ✅ Type Safety: Clear input/output contracts
        ✅ Composability: Can be combined with chains and agents
        ✅ Testing: Easier to test templates independently

        ❌ Trade-off: More abstraction overhead for simple cases
    ''')
    print(f"\n{benefits}")

    # Show validation
    print_subsection("Template Validation Example")
    try:
        # Missing required variable
        template.format()  # Will raise error
    except KeyError as e:
        print_output("Validation Error (expected)", str(e))

    print_subsection("When to Use What?")
    guidance = cleandoc('''
        Use F-Strings when:
        - Simple, one-off prompts
        - Few variables (<3)
        - No reuse needed
        - Maximum simplicity

        Use PromptTemplate when:
        - Reusing prompts across codebase
        - Need input validation
        - Combining with LangChain chains
        - Multiple template variations
    ''')
    print(f"\n{guidance}")


# endregion


# region 2. Partial Templates


def demo_partial_templates() -> None:
    """demonstrate partial template filling"""
    print_section("2. Partial Templates (Pre-filling Variables)")

    print_subsection("Scenario: Company Support Agent Template")

    # Create template with partial variables
    template = PromptTemplate(
        input_variables=["question"],
        partial_variables={"company": "Acme Corp", "tone": "friendly"},
        template=cleandoc('''
            You are a {tone} {company} support agent.

            User question: {question}

            Provide a helpful response:
        '''),
    )

    print_output("Template Structure", template.template)
    print_output("Required Variables", template.input_variables)
    print_output("Pre-filled Variables", template.partial_variables)

    # Use template with different questions
    print_subsection("Using Partial Template")
    questions = [
        "How do I reset my password?",
        "What are your business hours?",
        "Can I upgrade my plan?",
    ]

    for i, question in enumerate(questions, 1):
        prompt = template.format(question=question)
        print(f"\nExample {i}:")
        print(prompt)

    print_subsection("Why Use Partial Templates?")
    benefits = cleandoc('''
        ✅ DRY Principle: Define common variables once
        ✅ Consistency: Ensure consistent tone/branding
        ✅ Flexibility: Override partials when needed
        ✅ Configuration: Separate config from usage

        Perfect for:
        - Multi-tenant applications
        - Configurable agents
        - Role-based prompts
    ''')
    print(f"\n{benefits}")


# endregion


# region 3. ChatPromptTemplate


def demo_chat_templates() -> None:
    """demonstrate ChatPromptTemplate for multi-message prompts"""
    print_section("3. ChatPromptTemplate (Multi-Message Prompts)")

    print_subsection("Phase 2 Approach: Manual Message Construction")
    phase2_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Explain RAG"},
    ]
    print_output("Phase 2 Messages", phase2_messages)

    print_subsection("LangChain Approach: ChatPromptTemplate")
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a {role} assistant"),
        ("user", "Explain {topic}"),
    ])

    messages = template.format_messages(role="technical", topic="RAG")
    print_output("Template-Generated Messages", [msg.model_dump() for msg in messages])

    print_subsection("Advanced: System + Examples + User")
    advanced_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}"),
        ("human", "What is a vector database?"),
        ("ai", "A vector database stores high-dimensional embeddings..."),
        ("human", "{question}"),
    ])

    messages = advanced_template.format_messages(
        domain="AI systems",
        question="How do I choose the right embedding model?"
    )

    print("\nGenerated Conversation:")
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}]: {msg.content}")

    print_subsection("Why Use ChatPromptTemplate?")
    benefits = cleandoc('''
        ✅ Structure: Clear role-based message organization
        ✅ Few-Shot: Easy to add example conversations
        ✅ Validation: Ensures proper message format
        ✅ Integration: Works seamlessly with chat models

        Essential for:
        - Conversational agents
        - Role-based prompting
        - Few-shot learning
        - Chat history injection
    ''')
    print(f"\n{benefits}")


# endregion


# region 4. MessagesPlaceholder


def demo_messages_placeholder() -> None:
    """demonstrate MessagesPlaceholder for dynamic chat history"""
    print_section("4. MessagesPlaceholder (Dynamic Chat History)")

    print_subsection("Problem: Injecting Variable-Length Chat History")

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    print_output("Template Structure", template.messages)

    print_subsection("Example: Short Conversation")
    from langchain_core.messages import AIMessage, HumanMessage

    short_history = [
        HumanMessage(content="What is RAG?"),
        AIMessage(content="RAG is Retrieval Augmented Generation..."),
    ]

    messages = template.format_messages(
        chat_history=short_history,
        question="Can you give an example?"
    )

    print("\nGenerated Messages:")
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}]: {msg.content}")

    print_subsection("Example: Long Conversation")
    long_history = [
        HumanMessage(content="What is RAG?"),
        AIMessage(content="RAG is Retrieval Augmented Generation..."),
        HumanMessage(content="How does retrieval work?"),
        AIMessage(content="Retrieval uses vector similarity..."),
        HumanMessage(content="What about embedding models?"),
        AIMessage(content="Embedding models convert text to vectors..."),
    ]

    messages = template.format_messages(
        chat_history=long_history,
        question="Which embedding model should I use?"
    )

    print(f"\nGenerated {len(messages)} messages (system + {len(long_history)} history + 1 new)")

    print_subsection("Why Use MessagesPlaceholder?")
    benefits = cleandoc('''
        ✅ Dynamic Length: Handles variable chat history
        ✅ Type Safety: Ensures messages are proper format
        ✅ Flexibility: Works with any message type
        ✅ Integration: Perfect for chat memory systems

        Essential for:
        - Chatbots with memory
        - Conversational RAG
        - Multi-turn agents
        - Context-aware responses
    ''')
    print(f"\n{benefits}")


# endregion


# region 5. FewShotPromptTemplate


def demo_few_shot_templates() -> None:
    """demonstrate FewShotPromptTemplate for in-context learning"""
    print_section("5. FewShotPromptTemplate (In-Context Learning)")

    print_subsection("Scenario: Sentiment Analysis with Examples")

    # Define examples
    examples = [
        {"input": "I love this product!", "output": "positive"},
        {"input": "This is terrible.", "output": "negative"},
        {"input": "It's okay, not great.", "output": "neutral"},
    ]

    # Create example template
    example_template = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nSentiment: {output}",
    )

    # Create few-shot template
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Analyze the sentiment of the following text:",
        suffix="Input: {input}\nSentiment:",
        input_variables=["input"],
    )

    # Generate prompt
    prompt = few_shot_template.format(input="I really enjoy using this!")
    print_output("Generated Few-Shot Prompt", prompt)

    print_subsection("Dynamic Example Selection")
    from langchain_core.example_selectors import LengthBasedExampleSelector

    # Create selector that limits by token count
    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_template,
        max_length=100,  # approximate token limit
    )

    dynamic_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_template,
        prefix="Analyze the sentiment of the following text:",
        suffix="Input: {input}\nSentiment:",
        input_variables=["input"],
    )

    # Short input gets all examples
    short_prompt = dynamic_template.format(input="Great!")
    print_output("\nShort Input (gets all examples)", short_prompt)

    # Long input gets fewer examples (simulated)
    long_input = "This is a very long input " * 10
    long_prompt = dynamic_template.format(input=long_input)
    print_output("\nLong Input (automatic example reduction)", f"[{len(long_prompt)} chars]")

    print_subsection("Why Use FewShotPromptTemplate?")
    benefits = cleandoc('''
        ✅ In-Context Learning: Teach LLM through examples
        ✅ Consistency: Standardized example formatting
        ✅ Dynamic Selection: Auto-reduce examples for token limits
        ✅ Reusability: Share example sets across prompts

        Essential for:
        - Classification tasks
        - Structured output
        - Task-specific formatting
        - Consistent response patterns
    ''')
    print(f"\n{benefits}")


# endregion


# region 6. FewShotChatMessagePromptTemplate


def demo_few_shot_chat_templates() -> None:
    """demonstrate FewShotChatMessagePromptTemplate for chat-based few-shot"""
    print_section("6. FewShotChatMessagePromptTemplate (Chat Few-Shot)")

    print_subsection("Scenario: Teaching Response Style Through Examples")

    # Define example conversations
    examples = [
        {
            "input": "What is a database?",
            "output": "A database is a structured collection of data. Think of it as a digital filing cabinet where information is organized for easy retrieval.",
        },
        {
            "input": "What is an API?",
            "output": "An API (Application Programming Interface) is a set of rules that lets different software applications communicate. It's like a waiter taking your order to the kitchen and bringing back your food.",
        },
    ]

    # Create example template for each conversation turn
    example_template = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])

    # Create few-shot chat template
    few_shot_template = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_template,
    )

    # Create full template
    full_template = ChatPromptTemplate.from_messages([
        ("system", "You are a technical educator who explains concepts using analogies."),
        few_shot_template,
        ("human", "{input}"),
    ])

    # Generate messages
    messages = full_template.format_messages(input="What is a vector database?")

    print("\nGenerated Chat Messages:")
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"  [{role}]: {content}")

    print_subsection("Why Use FewShotChatMessagePromptTemplate?")
    benefits = cleandoc('''
        ✅ Response Style: Teach consistent response patterns
        ✅ Chat Format: Natural conversation examples
        ✅ Composability: Combines with ChatPromptTemplate
        ✅ Clarity: Clear separation of examples and query

        Essential for:
        - Teaching response tone/style
        - Consistent formatting patterns
        - Educational chatbots
        - Branded response voices
    ''')
    print(f"\n{benefits}")


# endregion


# region 7. Template Composition


def demo_template_composition() -> None:
    """demonstrate composing templates together"""
    print_section("7. Template Composition (Combining Templates)")

    print_subsection("Building Complex Prompts from Parts")

    # Component templates
    system_template = PromptTemplate.from_template(
        "You are an expert in {domain}"
    )

    context_template = PromptTemplate.from_template(cleandoc('''
        Context:
        {context}
    '''))

    question_template = PromptTemplate.from_template(
        "Question: {question}"
    )

    # Compose into chat template
    composed_template = ChatPromptTemplate.from_messages([
        ("system", system_template.format(domain="AI systems")),
        ("human", context_template.format(context="RAG uses vector similarity for retrieval")),
        ("human", question_template.format(question="How do I optimize retrieval?")),
    ])

    messages = composed_template.format_messages()

    print("\nComposed Messages:")
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        print(f"  [{role}]: {msg.content}")

    print_subsection("Template Pipelines")

    # Create reusable components
    rag_system = "You are a RAG system assistant"
    rag_instructions = cleandoc('''
        Instructions:
        1. Use the provided context
        2. Cite sources when possible
        3. Admit if information is not in context
    ''')

    # Pipeline template
    pipeline_template = ChatPromptTemplate.from_messages([
        ("system", rag_system),
        ("system", rag_instructions),
        MessagesPlaceholder(variable_name="context_messages"),
        ("human", "{question}"),
    ])

    print_output("\nPipeline Template Structure", pipeline_template.messages)

    print_subsection("Why Compose Templates?")
    benefits = cleandoc('''
        ✅ Modularity: Reuse components across prompts
        ✅ Maintainability: Update components independently
        ✅ Clarity: Clear separation of concerns
        ✅ Testing: Test components in isolation

        Essential for:
        - Complex prompt engineering
        - Multi-stage agents
        - Configurable systems
        - Large prompt libraries
    ''')
    print(f"\n{benefits}")


# endregion


# region 8. Decision Framework


def demo_decision_framework() -> None:
    """demonstrate when to use each template type"""
    print_section("8. Decision Framework: Choosing the Right Template")

    decision_tree = cleandoc('''
        ┌─ Need multi-message structure? (system/user/assistant)
        │  ├─ Yes ─> ChatPromptTemplate
        │  │         ├─ Need dynamic history? -> MessagesPlaceholder
        │  │         ├─ Need few-shot examples? -> FewShotChatMessagePromptTemplate
        │  │         └─ Simple conversation? -> ChatPromptTemplate.from_messages()
        │  │
        │  └─ No ─> Single string templates
        │            ├─ Need in-context examples? -> FewShotPromptTemplate
        │            ├─ Have common variables? -> PromptTemplate with partials
        │            ├─ Simple substitution? -> PromptTemplate.from_template()
        │            └─ One-off, <3 vars? -> F-String (Phase 2 approach)
        │
        └─ Complexity Rules:
           • Simple task, no reuse -> F-Strings
           • Reusable, validated -> PromptTemplate
           • Multi-turn chat -> ChatPromptTemplate
           • Teaching by example -> FewShotPromptTemplate
           • Complex systems -> Template composition
    ''')

    print(f"\n{decision_tree}")

    print_subsection("Real-World Examples")

    examples_map = {
        "Simple API call": "F-String",
        "Reusable analysis prompt": "PromptTemplate",
        "Support chatbot": "ChatPromptTemplate + MessagesPlaceholder",
        "Classification task": "FewShotPromptTemplate",
        "Educational chatbot": "FewShotChatMessagePromptTemplate",
        "RAG system": "ChatPromptTemplate + composition",
        "Multi-agent system": "Template composition + partials",
    }

    print("\nTask -> Template Recommendation:")
    for task, template in examples_map.items():
        print(f"  • {task:30} -> {template}")

    print_subsection("Trade-offs Summary")

    tradeoffs = cleandoc('''
        F-Strings:
        ✅ Simplicity, speed, familiarity
        ❌ No validation, hard to reuse, testing difficult

        PromptTemplate:
        ✅ Validation, reusability, testability
        ❌ Abstraction overhead, learning curve

        ChatPromptTemplate:
        ✅ Structured messages, role clarity, chat integration
        ❌ More complex, overkill for simple tasks

        FewShotPromptTemplate:
        ✅ In-context learning, consistency, dynamic examples
        ❌ Token usage, complexity, example management

        Composition:
        ✅ Modularity, flexibility, powerful
        ❌ High complexity, harder to debug
    ''')

    print(f"\n{tradeoffs}")


# endregion


# region Main


def main() -> None:
    """run all conceptual demonstrations"""
    print("\n" + "=" * 70)
    print("  LangChain Prompts & Templates - Conceptual Understanding")
    print("  No API key required - demonstrates patterns only")
    print("=" * 70)

    demo_basic_templates()
    demo_partial_templates()
    demo_chat_templates()
    demo_messages_placeholder()
    demo_few_shot_templates()
    demo_few_shot_chat_templates()
    demo_template_composition()
    demo_decision_framework()

    print("\n" + "=" * 70)
    print("  Conceptual demos complete!")
    print("  Next: Run practical.py with OPENAI_API_KEY for real LLM integration")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# endregion