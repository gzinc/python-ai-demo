"""
LangChain Prompts - Practical Examples with Real LLM Integration

This module demonstrates LangChain prompt templates with actual LLM API calls.
Requires OPENAI_API_KEY to be set in .env file.

Run: uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.practical

Requires: OPENAI_API_KEY in .env
"""

from inspect import cleandoc
from typing import TYPE_CHECKING

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.messages import AIMessage, HumanMessage

from phase7_frameworks.utils import check_api_keys, print_section, requires_openai

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


# region Helper Functions


def print_subsection(title: str) -> None:
    """print subsection header"""
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print('-' * 70)


def print_llm_output(label: str, response: str) -> None:
    """print LLM response with formatting"""
    print(f"\n{label}:")
    print(f"  {response}")


def get_llm(temperature: float = 0.7) -> "ChatOpenAI":
    """create configured LLM instance"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
    )


# endregion


# region 1. PromptTemplate with LLM


def demo_prompt_template_with_llm() -> None:
    """
    demonstrate PromptTemplate with actual LLM calls

    LCEL Pattern: Template | LLM
    ┌─────────────────────────────────────────────────────────────┐
    │         PromptTemplate with LLM Integration                 │
    │                                                             │
    │  1. Variable Input:                                         │
    │     {topic: "embeddings", style: "simple"}                  │
    │                    │                                        │
    │                    ▼                                        │
    │  2. PromptTemplate:                                         │
    │     ┌──────────────────────────────────────┐                │
    │     │ "Explain {topic} in {style} terms,   │                │
    │     │  using only 2 sentences."            │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │     Formatted: "Explain embeddings in simple terms,         │
    │                 using only 2 sentences."                    │
    │                    │                                        │
    │                    ▼                                        │
    │  3. LLM (gpt-4o-mini):                                      │
    │     ┌──────────────────────────────────────┐                │
    │     │  Processes formatted prompt          │                │
    │     │  Temperature: 0.3 (focused)          │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │  4. AIMessage Response:                                     │
    │     "Embeddings are numerical representations..."           │
    │                                                             │
    │  LCEL Syntax: chain = template | llm                        │
    │              response = chain.invoke(params)                │
    │                                                             │
    │  ✅ Benefit: Clean composition with pipe operator           │
    │  ✅ Benefit: Automatic prompt formatting                    │
    │  ✅ Benefit: Type-safe variable substitution                │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("1. PromptTemplate with Real LLM Integration")

    print_subsection("Simple Template -> LLM")

    # Create template
    template = PromptTemplate.from_template(
        "Explain {topic} in {style} terms, using only 2 sentences."
    )

    # Create LLM
    llm = get_llm(temperature=0.3)

    # Use with LCEL pipe operator
    chain = template | llm

    # Execute
    topics = [
        {"topic": "embeddings", "style": "simple"},
        {"topic": "vector databases", "style": "technical"},
        {"topic": "RAG systems", "style": "business"},
    ]

    for params in topics:
        print(f"\n{'─' * 70}")
        print(f"Topic: {params['topic']} | Style: {params['style']}")
        response = chain.invoke(params)
        print_llm_output("LLM Response", response.content)

    print_subsection("Key Pattern: LCEL Composition")
    explanation = cleandoc('''
        template | llm

        ✅ Pipe operator (|) chains components
        ✅ Template formats prompt
        ✅ LLM processes formatted prompt
        ✅ Returns AIMessage with response
    ''')
    print(f"\n{explanation}")


# endregion


# region 2. ChatPromptTemplate with LLM


def demo_chat_template_with_llm() -> None:
    """
    demonstrate ChatPromptTemplate with actual LLM calls

    Multi-Message Template Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │       ChatPromptTemplate: Multi-Message Conversations       │
    │                                                             │
    │  1. Input Variables:                                        │
    │    {domain: "machine learning", concept: "gradient descent"}│
    │                    │                                        │
    │                    ▼                                        │
    │  2. ChatPromptTemplate.from_messages():                     │
    │     ┌──────────────────────────────────────┐                │
    │     │ ("system", "You are an expert in     │                │
    │     │  {domain}. Provide concise answers") │                │
    │     ├──────────────────────────────────────┤                │
    │     │ ("human", "Explain {concept} in      │                │
    │     │  2-3 sentences.")                    │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │     Formatted Messages:                                     │
    │     [SystemMessage: "You are an expert in machine learning"]│
    │     [HumanMessage: "Explain gradient descent in 2-3..."]    │
    │                    │                                        │
    │                    ▼                                        │
    │  3. LLM Processing:                                         │
    │     ┌──────────────────────────────────────┐                │
    │     │  Chat model processes message list   │                │
    │     │  System message sets behavior        │                │
    │     │  Human message defines task          │                │
    │     └──────────────┬───────────────────────┘                │
    │                    │                                        │
    │                    ▼                                        │
    │  4. AIMessage Response:                                     │
    │     Expert-level explanation following system instruction   │
    │                                                             │
    │  ✅ Benefit: System message customizes LLM behavior         │
    │  ✅ Benefit: Supports multi-turn conversations              │
    │  ✅ Benefit: Clean separation of role-based messages        │
    └─────────────────────────────────────────────────────────────┘

    Few-Shot Pattern (Add Examples):
    ┌─────────────────────────────────────────────────────────────┐
    │                Teaching Response Style                      │
    │                                                             │
    │  Template Structure:                                        │
    │  ┌────────────────────────────────────────┐                 │
    │  │ ("system", "You are an educator...")   │                 │
    │  ├────────────────────────────────────────┤                 │
    │  │ ("human", "What is a cache?")          │  ← Example 1    │
    │  ├────────────────────────────────────────┤                 │
    │  │ ("ai", "A cache is like a desk...")    │  ← Response     │
    │  ├────────────────────────────────────────┤     style       │
    │  │ ("human", "What is {concept}?")        │  ← User input   │
    │  └────────────────────────────────────────┘                 │
    │                    │                                        │
    │                    ▼                                        │
    │  LLM learns tone, structure, and formatting from examples   │
    │  → Applies same style to new questions                      │
    │                                                             │
    │  ✅ Benefit: Consistent response formatting                 │
    │  ✅ Benefit: Easy to demonstrate desired tone               │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("2. ChatPromptTemplate with Real LLM Integration")

    print_subsection("Multi-Message Template -> LLM")

    # Create chat template
    template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert in {domain}. Always provide concise, accurate answers."),
        ("human", "Explain {concept} in 2-3 sentences."),
    ])

    # Create chain
    llm = get_llm(temperature=0.3)
    chain = template | llm

    # Execute with different domains
    queries = [
        {"domain": "machine learning", "concept": "gradient descent"},
        {"domain": "distributed systems", "concept": "eventual consistency"},
        {"domain": "frontend development", "concept": "virtual DOM"},
    ]

    for params in queries:
        print(f"\n{'─' * 70}")
        print(f"Domain: {params['domain']}")
        print(f"Concept: {params['concept']}")
        response = chain.invoke(params)
        print_llm_output("LLM Response", response.content)

    print_subsection("Adding Few-Shot Examples")

    # Template with examples
    expert_template = ChatPromptTemplate.from_messages([
        ("system", "You are a technical educator. Explain concepts with analogies."),
        ("human", "What is a cache?"),
        ("ai", "A cache is like a desk drawer. Instead of walking to the filing cabinet (slow storage) every time, you keep frequently used items in your drawer (fast cache) for quick access."),
        ("human", "What is {concept}?"),
    ])

    chain = expert_template | llm

    response = chain.invoke({"concept": "load balancer"})
    print(f"\n{'─' * 70}")
    print("Concept: load balancer")
    print_llm_output("LLM Response with Style", response.content)


# endregion


# region 3. MessagesPlaceholder with Chat History


def demo_messages_placeholder_with_llm() -> None:
    """
    demonstrate MessagesPlaceholder for chat memory

    Chat History Management Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │        MessagesPlaceholder: Conversational Context          │
    │                                                             │
    │  Template Structure:                                        │
    │  ┌────────────────────────────────────────┐                 │
    │  │ ("system", "You are an AI assistant")  │                 │
    │  ├────────────────────────────────────────┤                 │
    │  │ MessagesPlaceholder("chat_history")    │ ← Expands to    │
    │  ├────────────────────────────────────────┤   full history  │
    │  │ ("human", "{question}")                │                 │
    │  └────────────────────────────────────────┘                 │
    │                                                             │
    │  Conversation Flow:                                         │
    │  ┌──────────────────────────────────────────────┐           │
    │  │ Turn 1: "What is RAG?"                       │           │
    │  │   chat_history = []                          │           │
    │  │   LLM Response → Added to history            │           │
    │  │                                              │           │
    │  │ Turn 2: "Can you give me an example?"        │           │
    │  │   chat_history = [                           │           │
    │  │     HumanMessage("What is RAG?"),            │           │
    │  │     AIMessage("RAG is...")                   │           │
    │  │   ]                                          │           │
    │  │   LLM sees full context → coherent response  │           │
    │  │                                              │           │
    │  │ Turn 3: "What embedding model for that?"     │           │
    │  │   chat_history = [Turn 1 + Turn 2]           │           │
    │  │   LLM understands "that" refers to RAG       │           │
    │  └──────────────────────────────────────────────┘           │
    │                                                             │
    │  ✅ Benefit: Automatic context injection                    │
    │  ✅ Benefit: No manual prompt engineering                   │
    │  ✅ Benefit: Foundation for chat memory systems             │
    │  ✅ Benefit: Supports arbitrarily long conversations        │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("3. MessagesPlaceholder with Chat History")

    print_subsection("Building Conversational Context")

    # Create template with history placeholder
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Keep responses brief."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Create chain
    llm = get_llm(temperature=0.7)
    chain = template | llm

    # Simulate conversation
    chat_history = []

    print("\nConversation:")

    # Turn 1
    print(f"\n{'─' * 70}")
    print("User: What is RAG?")
    response = chain.invoke({
        "chat_history": chat_history,
        "question": "What is RAG?"
    })
    print_llm_output("AI", response.content)

    # Update history
    chat_history.extend([
        HumanMessage(content="What is RAG?"),
        AIMessage(content=response.content),
    ])

    # Turn 2 (references previous context)
    print(f"\n{'─' * 70}")
    print("User: Can you give me an example?")
    response = chain.invoke({
        "chat_history": chat_history,
        "question": "Can you give me an example?"
    })
    print_llm_output("AI", response.content)

    # Update history
    chat_history.extend([
        HumanMessage(content="Can you give me an example?"),
        AIMessage(content=response.content),
    ])

    # Turn 3 (continues context)
    print(f"\n{'─' * 70}")
    print("User: What embedding model should I use for that?")
    response = chain.invoke({
        "chat_history": chat_history,
        "question": "What embedding model should I use for that?"
    })
    print_llm_output("AI", response.content)

    print_subsection("Why This Works")
    explanation = cleandoc('''
        ✅ MessagesPlaceholder injects full conversation history
        ✅ LLM sees all previous context for coherent responses
        ✅ No manual prompt engineering for context
        ✅ Foundation for chat memory systems
    ''')
    print(f"\n{explanation}")


# endregion


# region 4. FewShotPromptTemplate with LLM


def demo_few_shot_with_llm() -> None:
    """
    demonstrate FewShotPromptTemplate for in-context learning

    Few-Shot Learning Pattern:
    ┌─────────────────────────────────────────────────────────────┐
    │        FewShotPromptTemplate: In-Context Learning           │
    │                                                             │
    │  Prompt Construction:                                       │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ PREFIX:                                        │         │
    │  │ "Classify sentiment as positive/negative..."   │         │
    │  ├────────────────────────────────────────────────┤         │
    │  │ EXAMPLES (teach format):                       │         │
    │  │   Text: "I loved this!"                        │         │
    │  │   Sentiment: positive                          │         │
    │  │                                                │         │
    │  │   Text: "Terrible quality"                     │         │
    │  │   Sentiment: negative                          │         │
    │  │                                                │         │
    │  │   Text: "It's okay"                            │         │
    │  │   Sentiment: neutral                           │         │
    │  ├────────────────────────────────────────────────┤         │
    │  │ SUFFIX (new input):                            │         │
    │  │   Text: {input}                                │         │
    │  │   Sentiment:                                   │         │
    │  └────────────────────────────────────────────────┘         │
    │                    │                                        │
    │                    ▼                                        │
    │  LLM learns pattern from examples                           │
    │                    │                                        │
    │                    ▼                                        │
    │  Consistent Output: "positive" | "negative" | "neutral"     │
    │                                                             │
    │  Benefits of Few-Shot Learning:                             │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ ✅ No fine-tuning required                     │         │
    │  │ ✅ Consistent output structure                 │         │
    │  │ ✅ Easy to update examples                     │         │
    │  │ ✅ Works with any classification task          │         │
    │  │ ✅ Temperature=0.0 for reproducibility         │         │
    │  └────────────────────────────────────────────────┘         │
    │                                                             │
    │  Use Cases:                                                 │
    │  • Sentiment analysis • Text classification                 │
    │  • Entity extraction • Format standardization               │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("4. FewShotPromptTemplate with In-Context Learning")

    print_subsection("Teaching Task Format Through Examples")

    # Define examples
    examples = [
        {
            "input": "I absolutely loved this product! Best purchase ever!",
            "output": "positive",
        },
        {
            "input": "Terrible quality. Waste of money.",
            "output": "negative",
        },
        {
            "input": "It's okay. Does what it says, nothing special.",
            "output": "neutral",
        },
    ]

    # Create example template
    example_template = PromptTemplate(
        input_variables=["input", "output"],
        template="Text: {input}\nSentiment: {output}",
    )

    # Create few-shot template
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        prefix="Classify the sentiment of the following text as positive, negative, or neutral:",
        suffix="Text: {input}\nSentiment:",
        input_variables=["input"],
    )

    # Create chain
    llm = get_llm(temperature=0.0)  # Low temperature for consistency
    chain = few_shot_template | llm

    # Test with new inputs
    test_texts = [
        "This exceeded my expectations! Highly recommend!",
        "Not worth the price. Very disappointed.",
        "It works as advertised. Nothing more, nothing less.",
        "Amazing! Would buy again in a heartbeat!",
    ]

    print("\nClassification Results:")
    for text in test_texts:
        response = chain.invoke({"input": text})
        # Extract just the sentiment (first word of response)
        sentiment = response.content.strip().split()[0]
        print(f"\n  Text: {text}")
        print(f"  Sentiment: {sentiment}")

    print_subsection("Why Few-Shot Works")
    explanation = cleandoc('''
        ✅ LLM learns format from examples
        ✅ Consistent output structure
        ✅ No fine-tuning required
        ✅ Easy to update examples
        ✅ Works with any classification task
    ''')
    print(f"\n{explanation}")


# endregion


# region 5. FewShotChatMessagePromptTemplate


def demo_few_shot_chat_with_llm() -> None:
    """
    demonstrate FewShotChatMessagePromptTemplate for response style

    Teaching Response Style with Chat Examples:
    ┌─────────────────────────────────────────────────────────────┐
    │     FewShotChatMessagePromptTemplate: Style Learning        │
    │                                                             │
    │  Full Template Structure:                                   │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ SYSTEM MESSAGE:                                │         │
    │  │ "You are a technical educator.                 │         │
    │  │  Follow response style from examples."         │         │
    │  ├────────────────────────────────────────────────┤         │
    │  │ EXAMPLE 1:                                     │         │
    │  │   Human: "What is a REST API?"                 │         │
    │  │   AI: "REST API is an interface...             │         │
    │  │                                                │         │
    │  │        🔑 Key Points:                         │         │
    │  │        • Uses HTTP methods                     │         │
    │  │        • Stateless communication               │         │
    │  │                                                │         │
    │  │        💡 Example: GET /users/123"             │         │
    │  ├────────────────────────────────────────────────┤         │
    │  │ EXAMPLE 2:                                     │         │
    │  │   Human: "What is GraphQL?"                    │         │
    │  │   AI: "GraphQL is a query language...          │         │
    │  │                                                │         │
    │  │        🔑 Key Points:                          │         │
    │  │        • Single endpoint                       │         │
    │  │        ...same structure..."                   │         │
    │  ├────────────────────────────────────────────────┤         │
    │  │ USER INPUT:                                    │         │
    │  │   Human: "{input}"                             │         │
    │  └────────────────────────────────────────────────┘         │
    │                    │                                        │
    │                    ▼                                        │
    │  LLM Pattern Learning:                                      │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ 1. Observe structure across examples           │         │
    │  │ 2. Extract tone and formatting patterns        │         │
    │  │ 3. Apply to new question                       │         │
    │  │ 4. Maintain consistent emoji usage             │         │
    │  │ 5. Follow example organization                 │         │
    │  └────────────────────────────────────────────────┘         │
    │                    │                                        │
    │                    ▼                                        │
    │  Styled Response matching example format                    │
    │                                                             │
    │  ✅ Benefit: Consistent branded responses                   │
    │  ✅ Benefit: Easy to update style                           │
    │  ✅ Benefit: Educational content standardization            │
    │  ✅ Benefit: Works for any response format                  │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("5. FewShotChatMessagePromptTemplate for Response Style")

    print_subsection("Teaching Response Pattern Through Examples")

    # Define example conversations
    examples = [
        {
            "input": "What is a REST API?",
            "output": cleandoc('''
                A REST API is an interface for communication between systems.

                🔑 Key Points:
                • Uses HTTP methods (GET, POST, PUT, DELETE)
                • Stateless communication
                • Resource-based URLs

                💡 Example: GET /users/123 retrieves user #123
            '''),
        },
        {
            "input": "What is GraphQL?",
            "output": cleandoc('''
                GraphQL is a query language for APIs that lets clients request exactly the data they need.

                🔑 Key Points:
                • Single endpoint for all queries
                • Client-defined response structure
                • Reduces over-fetching

                💡 Example: Query { user(id: 123) { name, email } }
            '''),
        },
    ]

    # Create example template
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
        ("system", "You are a technical educator. Follow the response style shown in examples."),
        few_shot_template,
        ("human", "{input}"),
    ])

    # Create chain
    llm = get_llm(temperature=0.7)
    chain = full_template | llm

    # Test with new question
    print("\nTeaching Response Style:")
    print(f"\n{'─' * 70}")
    print("Question: What is a WebSocket?")

    response = chain.invoke({"input": "What is a WebSocket?"})
    print_llm_output("Styled Response", response.content)

    print_subsection("Why This Pattern Works")
    explanation = cleandoc('''
        ✅ LLM learns tone and structure from examples
        ✅ Consistent formatting across responses
        ✅ Easy to update style by changing examples
        ✅ Perfect for branded or educational content
    ''')
    print(f"\n{explanation}")


# endregion


# region 6. Output Parsers


def demo_output_parsers() -> None:
    """
    demonstrate output parsers for structured responses

    Output Parser Pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │           Output Parsers: String → Structured Data          │
    │                                                             │
    │  Three Parser Types Demonstrated:                           │
    │                                                             │
    │  1. StrOutputParser (String):                               │
    │     Template | LLM | StrOutputParser()                      │
    │        │       │           │                                │
    │        └───────┴───────────┘                                │
    │                 │                                           │
    │                 ▼                                           │
    │     AIMessage.content → str (default extraction)            │
    │                                                             │
    │  2. CommaSeparatedListOutputParser (List):                  │
    │     Template | LLM | CommaSeparatedListOutputParser()       │
    │        │       │           │                                │
    │        └───────┴───────────┘                                │
    │                 │                                           │
    │                 ▼                                           │
    │     "Python, Java, Go" → ["Python", "Java", "Go"]           │
    │                                                             │
    │  3. JsonOutputParser (Structured):                          │
    │     Template | LLM | JsonOutputParser(pydantic_object)      │
    │        │       │           │                                │
    │        └───────┴───────────┘                                │
    │                 │                                           │
    │                 ▼                                           │
    │     JSON string → Dict[str, Any]                            │
    │                                                             │
    │  ┌─────────────────────────────────────────────┐            │
    │  │ Pydantic Schema Example:                    │            │
    │  │                                             │            │
    │  │ class TechStack(BaseModel):                 │            │
    │  │     frontend: str                           │            │
    │  │     backend: str                            │            │
    │  │     database: str                           │            │
    │  │     reason: str                             │            │
    │  │                                             │            │
    │  │ Parser validates against schema             │            │
    │  │ Returns typed dict                          │            │
    │  └─────────────────────────────────────────────┘            │
    │                                                             │
    │  Benefits of Output Parsers:                                │
    │  ✅ Type Safety: Structured data instead of strings         │
    │  ✅ Validation: Ensure LLM follows schema                   │
    │  ✅ Integration: Easy to use in application code            │
    │  ✅ Error Handling: Catch malformed responses early         │
    │  ✅ Documentation: Schema serves as API contract            │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("6. Output Parsers for Structured Data")

    print_subsection("String Output Parser (Default)")

    from langchain_core.output_parsers import StrOutputParser

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "List 3 benefits of {topic} (one per line, no numbering)"),
    ])

    llm = get_llm(temperature=0.3)
    chain = template | llm | StrOutputParser()

    response = chain.invoke({"topic": "vector databases"})
    print_llm_output("String Output", response)

    print_subsection("Comma-Separated List Parser")

    from langchain_core.output_parsers import CommaSeparatedListOutputParser

    list_parser = CommaSeparatedListOutputParser()

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "List 5 programming languages suitable for AI development"),
        ("human", "Format: comma-separated list only, no explanations"),
    ])

    chain = template | llm | list_parser

    languages = chain.invoke({})
    print(f"\nParsed List: {languages}")
    print(f"Type: {type(languages)}")

    print_subsection("Structured Output with JSON")

    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.pydantic_v1 import BaseModel, Field

    # Define structure
    class TechStack(BaseModel):
        """technology stack recommendation"""
        frontend: str = Field(description="frontend framework")
        backend: str = Field(description="backend framework")
        database: str = Field(description="database system")
        reason: str = Field(description="why this stack works well together")

    parser = JsonOutputParser(pydantic_object=TechStack)

    template = ChatPromptTemplate.from_messages([
        ("system", "You are a technical architect."),
        ("human", "Recommend a tech stack for a {app_type} application"),
        ("human", "Respond with JSON matching this format: {format_instructions}"),
    ])

    chain = template | llm | parser

    result = chain.invoke({
        "app_type": "real-time chat",
        "format_instructions": parser.get_format_instructions(),
    })

    print("\nStructured JSON Output:")
    import json
    print(json.dumps(result, indent=2))

    print_subsection("Why Use Output Parsers?")
    explanation = cleandoc('''
        ✅ Type Safety: Convert strings to structured data
        ✅ Validation: Ensure LLM output matches schema
        ✅ Integration: Easy to use parsed data in code
        ✅ Error Handling: Catch malformed responses
    ''')
    print(f"\n{explanation}")


# endregion


# region 7. Advanced: Partial Variables with Runtime Data


def demo_partial_with_runtime() -> None:
    """
    demonstrate partial variables with runtime-generated data

    Partial Variables: Dynamic Context Injection
    ┌─────────────────────────────────────────────────────────────┐
    │         Partial Variables with Runtime Data                 │
    │                                                             │
    │  Template Definition:                                       │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ PromptTemplate(                                │         │
    │  │   template="{context}\n\nQuestion: {question}",│         │
    │  │   input_variables=["question"],                │         │
    │  │   partial_variables={                          │         │
    │  │     "context": get_current_context  ←─────┐    │         │
    │  │   }                                       │    │         │
    │  │ )                                         │    │         │
    │  └───────────────────────────────────────────┼────┘         │
    │                                              │              │
    │                                              │              │
    │  Runtime Execution Flow:                     │              │
    │  ┌───────────────────────────────────────────┼────┐         │
    │  │ 1. User invokes chain:                    │    │         │
    │  │    chain.invoke({"question": "..."})      │    │         │
    │  │                                           │    │         │
    │  │ 2. Callable executed automatically:       │    │         │
    │  │    def get_current_context() -> str: ◄────┘    │         │
    │  │        return f"Current date: {now()}"         │         │
    │  │                                                │         │
    │  │ 3. Template formatted:                         │         │
    │  │    "Current date: 2026-01-18 14:30             │         │
    │  │                                                │         │
    │  │     Question: What day is it today?"           │         │
    │  │                                                │         │
    │  │ 4. LLM processes with fresh context            │         │
    │  └────────────────────────────────────────────────┘         │
    │                                                             │
    │  Use Cases for Partial Callables:                           │
    │  ┌────────────────────────────────────────────────┐         │
    │  │ • Current date/time (always fresh)             │         │
    │  │ • User session data (per-request)              │         │
    │  │ • System configuration (runtime values)        │         │
    │  │ • Environment context (deployment info)        │         │
    │  │ • Request metadata (headers, auth)             │         │
    │  └────────────────────────────────────────────────┘         │
    │                                                             │
    │  ✅ Benefit: DRY - reuse templates with dynamic data        │
    │  ✅ Benefit: Consistency - same logic across uses           │
    │  ✅ Benefit: Automatic - no manual context injection        │
    │  ✅ Benefit: Type-safe - callable signature validated       │
    └─────────────────────────────────────────────────────────────┘
    """
    print_section("7. Partial Variables with Runtime Context")

    print_subsection("Dynamic Context Injection")

    from datetime import datetime

    # Function to get current context
    def get_current_context() -> str:
        """generate runtime context"""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Create template with partial callable
    template = PromptTemplate(
        template=cleandoc('''
            {context}

            You are a helpful assistant.

            Question: {question}

            Provide a brief answer considering the current context:
        '''),
        input_variables=["question"],
        partial_variables={"context": get_current_context},
    )

    # Create chain
    llm = get_llm(temperature=0.7)
    chain = template | llm

    # Use template (context is auto-generated)
    print("\nQuestion: What day is it today?")
    response = chain.invoke({"question": "What day is it today?"})
    print_llm_output("Context-Aware Response", response.content)

    print_subsection("Why Use Partial Callables?")
    explanation = cleandoc('''
        ✅ Dynamic Data: Inject runtime information automatically
        ✅ DRY Principle: Reuse templates with changing context
        ✅ Consistency: Same context logic across all uses

        Use cases:
        • Current date/time
        • User session data
        • System configuration
        • Environment context
    ''')
    print(f"\n{explanation}")


# endregion


# region Main


def show_menu(has_openai: bool) -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  LangChain Prompts & Templates - Practical Examples")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "PromptTemplate with LLM", "template integration with real API calls", True),
        ("2", "ChatPromptTemplate with LLM", "multi-message templates in action", True),
        ("3", "MessagesPlaceholder with Chat History", "conversational context management", True),
        ("4", "FewShotPromptTemplate with LLM", "in-context learning demonstrations", True),
        ("5", "FewShotChatMessagePromptTemplate", "teaching response style", True),
        ("6", "Output Parsers", "structured data extraction", True),
        ("7", "Partial Variables with Runtime", "dynamic context injection", True),
    ]

    for num, name, desc, needs_api in demos:
        api_marker = "🔑" if needs_api else "  "
        status = "" if has_openai else " ⚠️ (needs API key)"
        print(f"  {api_marker} [{num}] {name}")
        print(f"      {desc}{status}")
        print()

    if not has_openai:
        print("  ⚠️  OpenAI API key required for all demos")
        print("     Set OPENAI_API_KEY in .env file")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str, has_openai: bool) -> bool:
    """run selected demos based on user input"""
    selections = selections.lower().strip()

    if selections == 'q':
        return False

    if not has_openai:
        print("\n⚠️  Cannot run demos: OPENAI_API_KEY not found")
        print("Please add OPENAI_API_KEY to your .env file")
        print("\nFor conceptual demos without API key, run:")
        print("  uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.concepts")
        return True

    demo_map = {
        '1': ("PromptTemplate with LLM", demo_prompt_template_with_llm),
        '2': ("ChatPromptTemplate with LLM", demo_chat_template_with_llm),
        '3': ("MessagesPlaceholder with Chat History", demo_messages_placeholder_with_llm),
        '4': ("FewShotPromptTemplate with LLM", demo_few_shot_with_llm),
        '5': ("FewShotChatMessagePromptTemplate", demo_few_shot_chat_with_llm),
        '6': ("Output Parsers", demo_output_parsers),
        '7': ("Partial Variables with Runtime", demo_partial_with_runtime),
    }

    if selections == 'a':
        # run all demos
        for name, demo_func in demo_map.values():
            try:
                demo_func()
            except Exception as e:
                print(f"\n❌ Error in {name}: {e}")
    else:
        # parse comma-separated selections
        selected = [s.strip() for s in selections.split(',')]
        for sel in selected:
            if sel in demo_map:
                name, demo_func = demo_map[sel]
                try:
                    demo_func()
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
            else:
                print(f"⚠️  Invalid selection: {sel}")

    return True


def main() -> None:
    """run demonstrations with interactive menu"""
    has_openai, _ = check_api_keys()

    print("\n" + "=" * 70)
    print("  LangChain Prompts & Templates - Practical Examples")
    print("  Using OpenAI API for real LLM integration")
    print("=" * 70)

    while True:
        show_menu(has_openai)
        selection = input("\nSelect demos to run (comma-separated) or 'a' for all: ").strip()

        if not selection:
            continue

        if not run_selected_demos(selection, has_openai):
            break

        print("\n" + "=" * 70)
        print("  Demos complete!")
        print("=" * 70)

        # pause before showing menu again
        try:
            input("\n⏸️  Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

    print("\n" + "=" * 70)
    print("  Thanks for exploring LangChain prompts!")
    print("  You now understand prompt templates with real LLM integration")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


# endregion
