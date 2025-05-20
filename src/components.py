import re
import string
import asyncio
from typing import Any
from openai import OpenAI
from prompt import *

class Node:
    def __init__(
        self,
        node_type,
        value = None,
        children = None,
        placeholder = None,
        is_grouped = False
    ):
        self.type = node_type          
        self.value = value              
        self.children = children
        self.placeholder = placeholder
        self.is_grouped = is_grouped
        
    def __repr__(self, level = 0):
        indent = '    ' * level
        if self.type == 'AtomicQuery':
            if self.placeholder:
                return f"{indent}AtomicQuery(value='{self.value}', placeholder={list(self.placeholder)})"
            else:
                return f"{indent}AtomicQuery(value='{self.value}')"
        else:
            child_reprs = ',\n'.join(child.__repr__(level + 1) for child in self.children)
            if self.value:
                if self.placeholder:
                    return f"{indent}{self.type}(value='{self.value}', placeholder={list(self.placeholder)}, [\n{child_reprs}\n{indent}])"
                else:
                    return f"{indent}{self.type}(value='{self.value}', [\n{child_reprs}\n{indent}])"
            else:
                if self.placeholder:
                    return f"{indent}{self.type}(placeholder={list(self.placeholder)}, [\n{child_reprs}\n{indent}])"
                else:
                    return f"{indent}{self.type}([\n{child_reprs}\n{indent}])"


class Translator:
    def __init__(
        self,
        client: OpenAI,
        model_name: str
    ):
        self.client = client
        self.model_name = model_name
    
    def change_model(
        self,
        model_name: str
    ):
        self.model_name = model_name
    
    async def translate(
        self,
        query: str,
        max_tries: int = 3,
        **kwargs
    ):
        user_prompt = f"question = {query}"
        
        messages = [
            {"role": "system", "content": BNF_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        for _ in range(max_tries):
            try:
                chat_response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model = self.model_name,
                    messages = messages,
                    **kwargs
                )
                
                return chat_response.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        raise RuntimeError("Failed to translate query after multiple attempts.")

# Parser 类
class Parser:
    def __init__(
        self
    ):
        self.tokens = None
        self.pos = 0
        self.letter_mapping = {}
        self.current_letter = None

    def tokenize(self, text):
        token_specification = [
            ('LPAREN',     r'\('),
            ('RPAREN',     r'\)'),
            ('PLUS',       r'\+'),
            ('TIMES',       r'\*'),
            ('SKIP',       r'\s+'),
            ('WORD',       r'[^+\*\(\)\)\s]+'),
            ('MISMATCH',   r'.'),
        ]
        tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_specification)
        get_token = re.compile(tok_regex).match
        pos = 0
        tokens = []
        mo = get_token(text)
        while mo is not None:
            kind = mo.lastgroup
            value = mo.group()
            if kind in ('WORD', 'PLUS', 'TIMES', 'LPAREN', 'RPAREN'):
                tokens.append((kind, value))
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected character {value!r} at position {pos}')
            pos = mo.end()
            mo = get_token(text, pos)
        tokens.append(('EOF', None))
        
        return tokens

    def _current_token(self):
        return self.tokens[self.pos]

    def _match(self, expected_type):
        if self._current_token()[0] == expected_type:
            self.pos += 1
        else:
            raise RuntimeError(f'Expected {expected_type}, got {self._current_token()}')

    def _replace_parentheses(self, text):

        pattern = re.compile(r'\(([^()+*]*)\)')
        while True:
            new_text, count = pattern.subn(r'[\1]', text)
            if count == 0:
                break 
            text = new_text 

        return text
    
    # 占位符处理
    def _extract_placeholders(self, query):
        # 使用正则表达式提取占位符
        placeholders = re.findall(r'\{(.*?)\}', query)
        return placeholders
    
    def _traverse_and_assign(self, node):
            if node.type == 'AtomicQuery':
                try:
                    letter = next(self.letter_iterator)
                except StopIteration:
                    raise RuntimeError("Exceeded letter assignments (A-Z).")
                
                self.letter_mapping[letter] = node.value
                return letter
            
            elif node.type in ('ListQuery', 'DependentQuery'):
                operator = ' + ' if node.type == 'ListQuery' else ' * '
                parts = []
                for child in node.children:
                    parts.append(self._traverse_and_assign(child))
                combined = operator.join(parts)
                if node.is_grouped:
                    return f"({combined})"
                else:
                    return combined
            else:
                raise RuntimeError(f'Unknown node type: {node.type}')
        

    def parse_complex_query(self, query: str):
        query = self._replace_parentheses(query)
        self.tokens = self.tokenize(query)
        self.pos = 0
        
        return self.parse_list_query()


    def get_letter_expression(self, parsed_tree: Node):
        self.letter_mapping = {}
        self.letter_iterator = iter(string.ascii_uppercase)
        expression = self._traverse_and_assign(parsed_tree)
        
        return expression, self.letter_mapping
    

    def parse_list_query(self):
        node = Node('ListQuery', children = [])
        child = self.parse_dependency_query()
        node.value = child.value
        node.children.append(child)
        
        while self._current_token()[0] == 'PLUS':
            self._match('PLUS')
            child = self.parse_dependency_query()
            node.children.append(child)
            node.value += ' + ' + child.value
        
        return node
    
    
    def parse_dependency_query(self):
        node = self.parse_atomic_query()
        
        while self._current_token()[0] == 'TIMES':
            self._match('TIMES')
            right = self.parse_atomic_query()
            node = Node('DependentQuery', children = [node, right], value = node.value + ' * ' + right.value)
            
        return node
    
    
    def parse_atomic_query(self):
        if self._current_token()[0] == 'WORD':
            words = []
            while self._current_token()[0] == 'WORD':
                words.append(self._current_token()[1])
                self._match('WORD')
            value = ' '.join(words)
            
            try:
                placeholder = self._extract_placeholders(value)
            except:
                placeholder = None
                
            value = value.replace('[', '(').replace(']', ')')
            return Node('AtomicQuery', value = value, placeholder = placeholder)
        
        elif self._current_token()[0] == 'LPAREN':
            self._match('LPAREN')
            node = self.parse_list_query()
            self._match('RPAREN')
            node.value = "(" + node.value + ")"
            node.is_grouped = True
            return node
        
        else:
            raise RuntimeError(f'Unexpected token {self._current_token()} at position {self.pos}')
    
    
    def is_valid_tree(self, node: Node, flag: bool = 0):
        
        if node.type == 'AtomicQuery':
            
            if flag == 0 and node.placeholder:
                print("false 1: independent query with placeholder")
                return False
            
            if flag == 1 and not node.placeholder:
                print("false 2: dependent query without placeholder")
                return False
            
            return True

        if node.type == 'DependentQuery':
            left, right = node.children[0], node.children[1]
            return self.is_valid_tree(left, flag = 0) and self.is_valid_tree(right, flag = 1)

        if node.type == 'ListQuery':
            for child in node.children:
                if not self.is_valid_tree(child, flag = flag):
                    print("false 3: list query with invalid child")
                    return False
            
            return True


class RecursiveDescentProcessor:
    def __init__(
        self,
        retriever,
        client: OpenAI,
        model_name: str
    ):
        self.retriever = retriever
        self.client = client
        self.model_name = model_name
    
    def change_model(
        self,
        model_name
    ):
        self.model_name = model_name
    
    def replace_placeholders(
        self,
        query: str,
        variables: dict
    ):
        def replacer(match):
            key = match.group(1)
            return str(variables.get(key, f"{{{key}}}"))
        new_query = re.sub(r'\{([^{}]+)\}', replacer, query)
        return new_query

    async def process_node(self, node: Node, context: str = None, topk: int = 1):
    
        if node.type == 'AtomicQuery':
            return await self._process_atomic_query(node, context, topk)
        elif node.type == 'DependentQuery':
            return await self._process_dependent_query(node, context, topk)
        elif node.type == 'ListQuery':
            return await self._process_list_query(node, context, topk)
        else:
            raise RuntimeError(f"Unknown node type '{node.type}'")

    async def _process_atomic_query(self, node: Node, context: str, topk: int):
        placeholder_dict = {}
        query = node.value
        placeholders = node.placeholder

        for placeholder in placeholders:
            
            placeholder_dict[placeholder] = await self._get_placeholder(
                query,
                context,
                placeholder,
                self.client,
                self.model_name
            )
            query = self.replace_placeholders(query, placeholder_dict)

        node.value = query

        documents = await self._atomic_search(
            retriever = self.retriever,
            query = query,
            topk = topk
        )
        
        titles = [doc['contents'].split("\n")[0] for doc in documents]
        documents = [doc['contents'].split('\n')[1] for doc in documents]
        
        doc_contents = "\n\n".join([f"Doc{index + 1}: {title}\n{doc}" for index, (title, doc) in enumerate(zip(titles, documents))])
        
        answer = await self._get_answer(
            query,
            doc_contents,
            self.client,
            self.model_name
        )
        sub_queries_results = f"query: {query}\nresult:{answer}\n"
        
        return [{"query": query, "documents": documents}], sub_queries_results

    async def _process_dependent_query(self, node: Node, context: str, topk: int):
        left_docs, left_answer = await self.process_node(node.children[0], context, topk)
        right_docs, right_answer = await self.process_node(node.children[1], left_answer, topk)

        sub_queries_results = f"{left_answer}\n{right_answer}\n\n"
        return left_docs + right_docs, sub_queries_results

    async def _process_list_query(self, node: Node, context: str, topk: int):
        sub_queries_results = ""
        all_docs = []
        for child in node.children:
            docs, sub_result = await self.process_node(child, context, topk)
            all_docs += docs
            sub_queries_results += sub_result
        return all_docs, sub_queries_results

    async def _get_placeholder(
        self,
        question: str,
        answer: str,
        placeholder: str,
        client: OpenAI,
        model_name: str
    ) -> str:
        system_prompt = GET_PLACEHOLDER_PROMPT
        user_prompt = f"Question: {question}\n Context: {answer}\nPlaceholder: {placeholder}\nAnswer: "
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = await asyncio.to_thread(
            client.chat.completions.create,
            model = model_name,
            messages = messages,
            temperature = 0.01,
            max_tokens = 256,
        )
        
        return chat_response.choices[0].message.content

    async def _get_answer(
        query: str,
        documents: str,
        client: OpenAI,
        model_name: str
    ) -> str:
        system_prompt = GET_ANSWER_PROMPT
        user_prompt = f"Documents:\n{documents}Question: {query}\n"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_response = await asyncio.to_thread(
            client.chat.completions.create,
            model = model_name,
            messages = messages,
            temperature = 0.01,
        )
        
        return chat_response.choices[0].message.content

    async def merge_answers(
        self,
        original_query: str,
        sub_queries_results: str,
    ) -> str:
        """
        Asynchronously merges answers from sub-queries into a single response using the OpenAI API.
        Args:
            original_query (str): The original query string.
            sub_queries_results (list): A list of results from sub-queries.
            client (OpenAI): An instance of the OpenAI client.
            model_name (str): The name of the model to use for generating the response.
        Returns:
            str: The merged response generated by the OpenAI model.
        """
        system_prompt = MERGE_ANSWER_PROMPT
        
        user_prompt = f"Original question: {original_query}\nSub-question and their answers:\n{sub_queries_results}\nOriginal question: {original_query}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        chat_response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model = self.model_name,
            messages = messages,
            temperature = 0.01
        )
        
        return chat_response.choices[0].message.content
    
    async def _atomic_search(
        self,
        retriever: Any,
        query: str,
        topk: int
    ):
        documents = await asyncio.to_thread(retriever.search, query, num = topk)
        return documents