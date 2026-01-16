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
    """demonstrate PromptTemplate with actual LLM calls"""
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
    """demonstrate ChatPromptTemplate with actual LLM calls"""
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
    """demonstrate MessagesPlaceholder for chat memory"""
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
    """demonstrate FewShotPromptTemplate for in-context learning"""
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
    """demonstrate FewShotChatMessagePromptTemplate for response style"""
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
    """demonstrate output parsers for structured responses"""
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
    """demonstrate partial variables with runtime-generated data"""
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


def main() -> None:
    """run all practical demonstrations"""
    has_openai, _ = check_api_keys()
    if not has_openai:
        print("\n" + "=" * 70)
        print("  ERROR: OPENAI_API_KEY not found in environment")
        print("=" * 70)
        print("\nThis module requires an OpenAI API key.")
        print("Please add OPENAI_API_KEY to your .env file.")
        print("\nFor conceptual demos without API key, run:")
        print("  uv run python -m phase7_frameworks.01_langchain_basics.01_prompts.concepts")
        print()
        return

    print("\n" + "=" * 70)
    print("  LangChain Prompts & Templates - Practical Examples")
    print("  Using OpenAI API for real LLM integration")
    print("=" * 70)

    demo_prompt_template_with_llm()
    demo_chat_template_with_llm()
    demo_messages_placeholder_with_llm()
    demo_few_shot_with_llm()
    demo_few_shot_chat_with_llm()
    demo_output_parsers()
    demo_partial_with_runtime()

    print("\n" + "=" * 70)
    print("  Practical demos complete!")
    print("  You now understand LangChain prompt templates with real LLM integration")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()


# endregion
