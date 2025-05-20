BNF_PROMPT = """You are an expert in query intent understanding, tasked with decomposing complex queries into basic components. Follow the step-by-step procedure below to generate a BNF-compliant expression for each query.

==================================================
 Query Types and Grammar Definitions
==================================================

Query Types:
    1. AtomicQuery:
       • Simple, direct queries that require a factual answer.
       • Non-decomposable, orthogonal, and non-redundant.

    2. DependentQuery:
       • Multi-step queries where each step depends on the result of the previous one.
       • Composed of multiple AtomicQueries with dependencies.

    3. ListQuery:
       • Requires decomposition into multiple parallel, independent sub-queries.

BNF Definitions:
    • Set of atomic query terms (W): 
         All possible atomic query strings, where each atomic query is independent and non-redundant.

    • Set of operators (O): {'+' (parallel), '*' (dependent)}

    • <AtomicQuery> ::= w ∈ W | '(' <ListQuery> ')'
        - expressions enclosed in parentheses can also be considered as a production rule for <AtomicQuery>.

    • <DependentQuery> ::= <AtomicQuery> | <DependentQuery> '*' <AtomicQuery>
        - <AtomicQuery> may include placeholders formatted as {placeholder name}.
        - '*' indicates that the next query depends on the result of the previous query.

    • <ListQuery> ::= <DependentQuery> | <ListQuery> '+' <DependentQuery>
        - '+' denotes parallel relationships among queries.

==================================================
 Task Instructions
==================================================
1. Decompose the user's query into an ordered set of AtomicQueries, ensuring each query is factual, independent, and non-redundant.
2. Determine whether sub-queries are parallel (use '+') or dependent (use '*').
3. If you use '*', the next query must have a placeholder referencing the previous step’s result, e.g., {placeholder}.
4. Use parentheses '()' to group sub-queries when necessary.
5. Maintain the same language as the input query when formulating AtomicQueries.

Please decompose and compile each complex query into a BNF-compliant expression using '+', '*', '()', and '{placeholders}', then output in the specified format.
"""

GET_PLACEHOLDER_PROMPT = """
Extract the value of placeholder based on the questions and answers before, usually return with just an entity.

Example:
Question: Who captured Malakoff
Context: French troops under the command of Marshal Pelissier, later the Duke of Malakoff (French: Duc de Malakoff), and General Patrice de Mac-Mahon captured Malakoff.
Placeholder: capturers
Answer: French troops

Please extract the value of the placeholder based on the given question and answer. Only give me the value of the placeholder and do not output any other words.
"""

GET_ANSWER_PROMPT = """
Answer the question based on the given documents, dont output any other words. The following are the given question and documents.

"""

MERGE_ANSWER_PROMPT = """
Answer the original question based on the sub-questions and their answers (dont answer with any explaination). Only give me the answer to the original question and do not output any other words.
"""