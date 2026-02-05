"""
Chains - Conceptual Overview (No API Key Required)

Demonstrates LangChain chain composition patterns, LCEL syntax, and chain architecture
without making actual LLM calls.

Run: uv run python -m phase7_frameworks.01_langchain_basics.03_chains.concepts
"""

from inspect import cleandoc


def print_section(title: str) -> None:
    """print section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# region Demo 1: Chain Concept Overview


def demo_chain_concept() -> None:
    """explain what chains are and why they matter"""
    print_section("Demo 1: Chain Concept Overview")

    print(cleandoc('''
        ## What is a Chain?

        A chain connects multiple components into a unified workflow:

        ```
        Input → [Component 1] → [Component 2] → [Component 3] → Output
        ```

        ## Why Chains?

        **Composability**: Build complex workflows from simple components
        - Each component does one thing well
        - Components can be reused across different chains
        - Easy to modify and extend

        **Maintainability**: Modify components independently
        - Change prompt without touching LLM code
        - Swap LLM provider without rewriting logic
        - Add new processing steps without breaking existing flow

        **Observability**: Track execution and debug
        - See intermediate results
        - Identify bottlenecks
        - Debug failures systematically

        ## Example Workflow

        ```
        User Input: "Explain quantum computing"

        Step 1 [Prompt]: Format with template
        → "You are a teacher. Explain quantum computing simply."

        Step 2 [LLM]: Generate response
        → "Quantum computing uses quantum mechanics to..."

        Step 3 [Parser]: Extract clean text
        → "Quantum computing uses quantum mechanics to..."

        Output: Clean string result
        ```

        ## Chain Evolution

        LangChain chains evolved through 3 generations:

        **Gen 1**: LLMChain (legacy, verbose)
        **Gen 2**: SequentialChain (multi-step, rigid)
        **Gen 3**: LCEL (modern, flexible, recommended)
    '''))


# endregion

# region Demo 2: LLMChain Pattern (Legacy)


def demo_llmchain_pattern() -> None:
    """show legacy LLMChain pattern"""
    print_section("Demo 2: LLMChain Pattern (Legacy)")

    print(cleandoc('''
        ## LLMChain Pattern (Pre-LCEL)

        Classic approach for simple prompt → LLM workflows:

        ```python
        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import ChatOpenAI

        # Components
        prompt = PromptTemplate.from_template("Explain {concept}")
        llm = ChatOpenAI(model="gpt-3.5-turbo")

        # Chain creation
        chain = LLMChain(llm=llm, prompt=prompt)

        # Execution
        result = chain.run(concept="embeddings")
        ```

        ## Characteristics

        **Pros**:
        - Simple for basic use cases
        - Well-documented (older docs)
        - Backwards compatible

        **Cons**:
        - Verbose syntax
        - Limited composition
        - No built-in streaming
        - No async support
        - Hard to debug

        ## When to Use

        - Legacy code maintenance
        - Following old tutorials
        - Simple single-step operations

        **Recommendation**: Use LCEL for all new code
    '''))

    print("\n## Code Comparison:\n")

    comparison = {
        "Legacy (LLMChain)": cleandoc('''
            chain = LLMChain(llm=llm, prompt=prompt)
            result = chain.run(concept="embeddings")
        '''),
        "Modern (LCEL)": cleandoc('''
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"concept": "embeddings"})
        ''')
    }

    for approach, code in comparison.items():
        print(f"**{approach}**:")
        print(code)
        print()


# endregion

# region Demo 3: SequentialChain Pattern


def demo_sequential_chain() -> None:
    """show sequential chain for multi-step operations"""
    print_section("Demo 3: SequentialChain Pattern")

    print(cleandoc('''
        ## SequentialChain Pattern

        Chains multiple LLMChains in sequence:

        ```python
        from langchain.chains import SequentialChain

        # Chain 1: Summarize text
        summarize_chain = LLMChain(
            llm=llm,
            prompt=summarize_prompt,
            output_key="summary"
        )

        # Chain 2: Extract keywords from summary
        keyword_chain = LLMChain(
            llm=llm,
            prompt=keyword_prompt,
            output_key="keywords"
        )

        # Chain 3: Analyze sentiment
        sentiment_chain = LLMChain(
            llm=llm,
            prompt=sentiment_prompt,
            output_key="sentiment"
        )

        # Compose sequential chain
        chain = SequentialChain(
            chains=[summarize_chain, keyword_chain, sentiment_chain],
            input_variables=["text"],
            output_variables=["summary", "keywords", "sentiment"]
        )

        # Execute
        result = chain({"text": "Long article text..."})
        # Output: {"summary": "...", "keywords": [...], "sentiment": "positive"}
        ```

        ## Characteristics

        **Pros**:
        - Clear multi-step logic
        - Explicit variable passing
        - Good for complex pipelines

        **Cons**:
        - Rigid linear flow (no branching)
        - All-or-nothing execution
        - Limited error recovery
        - Verbose setup

        ## Use Cases

        - Multi-step analysis pipelines
        - Document processing workflows
        - Data transformation sequences
        - ETL-like operations with LLMs
    '''))

    print("\n## Example Flow:\n")

    flow_steps = [
        ("Input", '{"text": "Article about AI advances..."}'),
        ("Step 1: Summarize", '{"summary": "AI is advancing rapidly..."}'),
        ("Step 2: Keywords", '{"keywords": ["AI", "machine learning", "progress"]}'),
        ("Step 3: Sentiment", '{"sentiment": "optimistic"}'),
        ("Output", '{"summary": "...", "keywords": [...], "sentiment": "optimistic"}')
    ]

    for step, result in flow_steps:
        print(f"  {step:20s} → {result}")


# endregion

# region Demo 4: LCEL Syntax (Modern)


def demo_lcel_syntax() -> None:
    """demonstrate modern LCEL pipe operator syntax"""
    print_section("Demo 4: LCEL Syntax (Modern Standard)")

    print(cleandoc('''
        ## LCEL (LangChain Expression Language)

        Modern chain composition using the pipe operator (|):

        ```python
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_openai import ChatOpenAI

        # Components
        prompt = ChatPromptTemplate.from_template("Explain {concept}")
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        parser = StrOutputParser()

        # LCEL composition (Unix pipe style)
        chain = prompt | llm | parser

        # Execution
        result = chain.invoke({"concept": "embeddings"})
        ```

        ## Why LCEL?

        **Concise Syntax**: 3 components = 1 line
        - No boilerplate code
        - Reads left-to-right like Unix pipes
        - Clear data flow

        **Built-in Streaming**: Automatic support
        - for chunk in chain.stream(input)
        - Progressive results
        - Better UX

        **Native Async**: First-class async/await
        - await chain.ainvoke(input)
        - Parallel execution
        - High concurrency

        **Advanced Features**: Retries, fallbacks, batching
        - chain.with_retry()
        - chain.with_fallbacks([fallback])
        - chain.batch([input1, input2])

        ## Runnable Interface

        All LCEL components implement Runnable:

        ```python
        # Synchronous
        result = runnable.invoke(input)           # Single
        results = runnable.batch([input1, input2])  # Batch
        for chunk in runnable.stream(input):      # Stream
            print(chunk)

        # Asynchronous
        result = await runnable.ainvoke(input)
        results = await runnable.abatch([...])
        async for chunk in runnable.astream(input):
            print(chunk)
        ```
    '''))

    print("\n## LCEL vs Legacy Comparison:\n")

    comparison_table = [
        ("Feature", "Legacy (LLMChain)", "LCEL"),
        ("-" * 20, "-" * 30, "-" * 30),
        ("Syntax", "Verbose (3-5 lines)", "Concise (1 line)"),
        ("Streaming", "Manual setup", "Built-in"),
        ("Async", "Limited", "Native support"),
        ("Composition", "Complex", "Simple (| operator)"),
        ("Debugging", "Basic", "Advanced tracing"),
        ("Recommendation", "Legacy only", "Use for all new code")
    ]

    for feature, legacy, lcel in comparison_table:
        print(f"  {feature:20s} | {legacy:30s} | {lcel}")


# endregion

# region Demo 5: LCEL Components


def demo_lcel_components() -> None:
    """show common LCEL runnable components"""
    print_section("Demo 5: Common LCEL Components")

    print(cleandoc('''
        ## Core Runnable Types

        All LCEL components implement the Runnable interface with invoke/stream/batch methods.

        ### 1. ChatPromptTemplate
        ```python
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant"),
            ("human", "{input}")
        ])
        # Usage: prompt.invoke({"input": "Hello"})
        ```

        ### 2. Chat Models (LLMs)
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic

        llm = ChatOpenAI(model="gpt-3.5-turbo")
        # Usage: llm.invoke(messages)
        ```

        ### 3. Output Parsers
        ```python
        from langchain_core.output_parsers import StrOutputParser

        parser = StrOutputParser()  # Extracts content string
        # Usage: parser.invoke(ai_message)
        ```

        ### 4. RunnablePassthrough
        ```python
        from langchain_core.runnables import RunnablePassthrough

        passthrough = RunnablePassthrough()  # Passes data unchanged
        # Usage: For side information in chains
        ```

        ### 5. RunnableLambda
        ```python
        from langchain_core.runnables import RunnableLambda

        def process(x):
            return x.upper()

        processor = RunnableLambda(process)
        # Usage: Custom transformations in chains
        ```

        ### 6. RunnableParallel
        ```python
        from langchain_core.runnables import RunnableParallel

        parallel = RunnableParallel({
            "summary": summary_chain,
            "keywords": keyword_chain
        })
        # Usage: Execute multiple chains in parallel
        ```
    '''))

    print("\n## Component Chaining Examples:\n")

    examples = {
        "Basic": "prompt | llm | parser",
        "With Passthrough": "{\"context\": retriever, \"question\": RunnablePassthrough()} | prompt | llm",
        "With Lambda": "prompt | llm | RunnableLambda(lambda x: x.upper())",
        "Parallel": "RunnableParallel({\"a\": chain1, \"b\": chain2})",
    }

    for name, example in examples.items():
        print(f"  {name:15s}: {example}")


# endregion

# region Demo 6: LCEL Patterns


def demo_lcel_patterns() -> None:
    """show common LCEL composition patterns"""
    print_section("Demo 6: Common LCEL Patterns")

    print(cleandoc('''
        ## Pattern 1: Basic Linear Chain

        Simplest pattern - sequential components:

        ```python
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"concept": "RAG"})
        ```

        Flow: Input → Prompt → LLM → Parser → Output

        ---

        ## Pattern 2: Parallel Execution

        Execute multiple chains simultaneously:

        ```python
        from langchain_core.runnables import RunnableParallel

        chain = RunnableParallel({
            "summary": prompt1 | llm | parser,
            "keywords": prompt2 | llm | parser,
            "sentiment": prompt3 | llm | parser
        })

        result = chain.invoke({"text": "Article..."})
        # {"summary": "...", "keywords": [...], "sentiment": "positive"}
        ```

        Flow: Input → [Chain1, Chain2, Chain3] → Combined Output

        ---

        ## Pattern 3: Branching Logic

        Conditional execution based on input:

        ```python
        from langchain_core.runnables import RunnableBranch

        chain = RunnableBranch(
            (lambda x: len(x["text"]) < 100, short_chain),
            (lambda x: len(x["text"]) < 1000, medium_chain),
            long_chain  # default
        )
        ```

        Flow: Input → Condition Check → Selected Chain → Output

        ---

        ## Pattern 4: Passthrough + Side Information

        Pass input alongside processed data:

        ```python
        from langchain_core.runnables import RunnablePassthrough

        chain = {
            "context": retriever | format_docs,  # Process
            "question": RunnablePassthrough()    # Pass through
        } | prompt | llm | parser

        # Input: {"question": "What is RAG?"}
        # Prompt receives: {"context": "...", "question": "What is RAG?"}
        ```

        Use Case: RAG systems where you need both retrieved context and original question

        ---

        ## Pattern 5: Retry with Fallback

        Handle failures gracefully:

        ```python
        primary = prompt | gpt4_llm | parser
        fallback = prompt | gpt35_llm | parser

        chain = primary.with_fallbacks([fallback])
        ```

        Flow: Try Primary → If Fail → Try Fallback → If Fail → Raise Error
    '''))

    print("\n## Pattern Comparison:\n")

    patterns = [
        ("Pattern", "Use Case", "Complexity"),
        ("-" * 20, "-" * 35, "-" * 10),
        ("Linear", "Simple workflows", "Low"),
        ("Parallel", "Independent operations", "Medium"),
        ("Branching", "Conditional logic", "Medium"),
        ("Passthrough", "RAG, context + query", "Medium"),
        ("Fallback", "Reliability, error handling", "Low")
    ]

    for pattern, use_case, complexity in patterns:
        print(f"  {pattern:20s} | {use_case:35s} | {complexity}")


# endregion

# region Demo 7: Error Handling


def demo_error_handling() -> None:
    """show error handling strategies in chains"""
    print_section("Demo 7: Error Handling in Chains")

    print(cleandoc('''
        ## Strategy 1: Try/Except (Manual)

        Basic error handling:

        ```python
        try:
            result = chain.invoke({"input": "..."})
        except Exception as e:
            print(f"Chain failed: {e}")
            # Fallback logic or logging
        ```

        **Use When**: Simple error recovery, custom fallback logic

        ---

        ## Strategy 2: Fallback Chains (Automatic)

        Automatic provider/model switching:

        ```python
        primary = prompt | ChatOpenAI(model="gpt-4") | parser
        fallback1 = prompt | ChatAnthropic(model="claude-3") | parser
        fallback2 = prompt | ChatOpenAI(model="gpt-3.5-turbo") | parser

        chain = primary.with_fallbacks([fallback1, fallback2])
        ```

        **Execution**:
        1. Try GPT-4
        2. If fails → Try Claude-3
        3. If fails → Try GPT-3.5
        4. If fails → Raise exception

        **Use When**: Provider reliability, cost optimization, model availability

        ---

        ## Strategy 3: Retry Configuration

        Automatic retries with exponential backoff:

        ```python
        from langchain_core.runnables import RunnableRetry

        chain = (prompt | llm | parser).with_retry(
            stop_after_attempt=3,           # Max 3 tries
            wait_exponential_multiplier=1,  # Wait 1, 2, 4 seconds
            wait_exponential_max=10,        # Max 10 seconds
            retry_if_exception_type=(TimeoutError, RateLimitError)
        )
        ```

        **Retry Schedule**:
        - Attempt 1: Immediate
        - Attempt 2: Wait 1 second
        - Attempt 3: Wait 2 seconds
        - Attempt 4: Wait 4 seconds (if max_attempts > 3)

        **Use When**: Transient failures (timeouts, rate limits, network issues)

        ---

        ## Strategy 4: Combining Strategies

        Retries + fallbacks for maximum reliability:

        ```python
        # Each chain has retries
        primary = (prompt | gpt4_llm | parser).with_retry(stop_after_attempt=2)
        fallback = (prompt | gpt35_llm | parser).with_retry(stop_after_attempt=2)

        # Fallback between them
        chain = primary.with_fallbacks([fallback])
        ```

        **Execution Flow**:
        1. Try GPT-4 (up to 2 retries)
        2. If all retries fail → Try GPT-3.5 (up to 2 retries)
        3. If all fail → Raise exception

        Total attempts: Up to 4 (2 per model)
    '''))

    print("\n## Error Handling Decision Tree:\n")

    decision_tree = cleandoc('''
        Error Type                     → Strategy
        ─────────────────────────────────────────────────────────
        Transient (timeout, rate)      → Retry with backoff
        Provider outage                → Fallback to alternative
        Model unavailable              → Fallback to different model
        Invalid input                  → Try/except with validation
        Cost optimization              → Fallback to cheaper model
        Maximum reliability            → Retry + Fallback combined
    ''')

    print(decision_tree)


# endregion

# region Demo 8: Debugging Chains


def demo_debugging() -> None:
    """show techniques for debugging chain execution"""
    print_section("Demo 8: Debugging Chain Execution")

    print(cleandoc('''
        ## Technique 1: Verbose Mode

        Enable detailed execution logging:

        ```python
        chain = prompt | llm | parser
        result = chain.invoke(
            {"concept": "embeddings"},
            config={"verbose": True}
        )
        ```

        **Output**: Shows each component's input/output during execution

        ---

        ## Technique 2: Intermediate Steps

        Inspect data between components:

        ```python
        from langchain_core.runnables import RunnablePassthrough

        def debug_print(x):
            print(f"[DEBUG] Data: {x}")
            return x

        chain = (
            prompt
            | RunnableLambda(debug_print)  # Inspect after prompt
            | llm
            | RunnableLambda(debug_print)  # Inspect after LLM
            | parser
        )
        ```

        ---

        ## Technique 3: Component Testing

        Test individual components:

        ```python
        # Test prompt
        formatted = prompt.invoke({"concept": "embeddings"})
        print(f"Prompt output: {formatted}")

        # Test LLM
        llm_output = llm.invoke(formatted)
        print(f"LLM output: {llm_output}")

        # Test parser
        parsed = parser.invoke(llm_output)
        print(f"Parser output: {parsed}")
        ```

        ---

        ## Technique 4: LangSmith Tracing (Production)

        Cloud-based observability platform:

        ```python
        import os

        # Enable tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"

        # All chain executions now traced
        result = chain.invoke({"concept": "embeddings"})
        ```

        **Features**:
        - Visual execution graphs
        - Token usage tracking
        - Latency analysis
        - Error tracking
        - Performance optimization insights

        **Access**: https://smith.langchain.com/

        ---

        ## Technique 5: Custom Callbacks

        Hook into chain lifecycle:

        ```python
        from langchain.callbacks import StdOutCallbackHandler

        chain = prompt | llm | parser
        result = chain.invoke(
            {"concept": "embeddings"},
            config={"callbacks": [StdOutCallbackHandler()]}
        )
        ```

        **Lifecycle Events**:
        - on_chain_start
        - on_llm_start
        - on_llm_end
        - on_chain_end
        - on_chain_error
    '''))

    print("\n## Debugging Workflow:\n")

    workflow = [
        ("1. Quick Check", "Run with verbose=True"),
        ("2. Component Test", "Test each component in isolation"),
        ("3. Inspect Data", "Add RunnableLambda debug points"),
        ("4. Production Monitor", "Enable LangSmith tracing"),
        ("5. Deep Dive", "Custom callbacks for specific events")
    ]

    for step, action in workflow:
        print(f"  {step:20s} → {action}")


# endregion


def show_menu() -> None:
    """display interactive demo menu"""
    print("\n" + "=" * 70)
    print("  Chains - Conceptual Examples")
    print("=" * 70)
    print("\n📚 Available Demos:\n")

    demos = [
        ("1", "Chain Concept Overview", "what chains are and why they matter"),
        ("2", "LLMChain Pattern (Legacy)", "traditional chain approach"),
        ("3", "SequentialChain Pattern", "multi-step linear workflows"),
        ("4", "LCEL Syntax (Modern)", "pipe operator composition"),
        ("5", "LCEL Components", "runnable types and interfaces"),
        ("6", "LCEL Patterns", "common composition patterns"),
        ("7", "Error Handling", "strategies for chain failures"),
        ("8", "Debugging Chains", "techniques for troubleshooting"),
    ]

    for num, name, desc in demos:
        print(f"    [{num}] {name}")
        print(f"        {desc}")
        print()

    print("  [a] Run all demos")
    print("  [q] Quit")
    print("\n" + "=" * 70)


def run_selected_demos(selections: str) -> bool:
    """run selected demos based on user input"""
    selections = selections.lower().strip()

    if selections == 'q':
        return False

    demo_map = {
        '1': ("Chain Concept Overview", demo_chain_concept),
        '2': ("LLMChain Pattern", demo_llmchain_pattern),
        '3': ("SequentialChain Pattern", demo_sequential_chain),
        '4': ("LCEL Syntax", demo_lcel_syntax),
        '5': ("LCEL Components", demo_lcel_components),
        '6': ("LCEL Patterns", demo_lcel_patterns),
        '7': ("Error Handling", demo_error_handling),
        '8': ("Debugging Chains", demo_debugging),
    }

    if selections == 'a':
        # run all demos
        for name, demo_func in demo_map.values():
            demo_func()
    else:
        # parse comma-separated selections
        selected = [s.strip() for s in selections.split(',')]
        for sel in selected:
            if sel in demo_map:
                name, demo_func = demo_map[sel]
                demo_func()
            else:
                print(f"⚠️  Invalid selection: {sel}")

    return True


def main() -> None:
    """run demonstrations with interactive menu"""
    print("\n" + "=" * 70)
    print("  Chains - Conceptual Understanding")
    print("  No API key required - demonstrates patterns only")
    print("=" * 70)

    while True:
        show_menu()
        selection = input("\nSelect demos to run (comma-separated) or 'a' for all: ").strip()

        if not selection:
            continue

        if not run_selected_demos(selection):
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
    print("  Thanks for exploring LangChain chains!")
    print("  Next: Run practical.py for hands-on practice with real LLM calls")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")