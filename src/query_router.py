from typing import Dict, Any, Literal
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re


class QueryRouter:
    
    def __init__(self, llm_model = 'mistral'):
        self.llm = OllamaLLM(model = llm_model, temperature=0.0)
        self.build_classifier()
        print(f"Query Router Initialized with model : {llm_model}")
        
        
    def build_classifier(self):
        
        classification_template = """You are a query classification expert for a sales forecasting and business intelligence system.

                                    Classify the following user query into ONE of these categories:

                                    1. SALES - Queries about numerical sales data, predictions, comparisons, totals, averages, trends
                                    Examples:
                                    - "What will be the sales of Kathmandu Mart next year?"
                                    - "Compare sales between 2024 and 2025"
                                    - "What's the average weekly sales for Electronics?"
                                    - "Total sales for Bhat-Bhateni in Q1 2025"

                                    2. CONTEXTUAL - Queries about holidays, events, context, reasons, explanations
                                    Examples:
                                    - "What holidays are in October?"
                                    - "Why do sales increase during Dashain?"
                                    - "Tell me about Tihar festival"
                                    - "When is Dashain celebrated?"

                                    3. HYBRID - Queries that need BOTH sales data AND contextual information
                                    Examples:
                                    - "How do Dashain holidays affect sales?"
                                    - "What's the sales impact during major festivals?"
                                    - "Compare sales during holiday vs non-holiday periods"

                                    Respond with ONLY ONE WORD: SALES, CONTEXTUAL, or HYBRID

                                    Query: {query}

                                    Classification:"""
        
        self.prompt = PromptTemplate.from_template(classification_template)
        self.classification_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def classify_query(self, query):
        
        try:        
            result = self.classification_chain.invoke({'query': query})
            result = result.strip().upper()
            
            #validating the calssification word
            
            if 'SALES' in result:
                return 'SALES'
            
            elif 'CONTEXTUAL' in result:
                return 'CONTEXTUAL'
            
            elif 'HYBRID' in result:
                return 'HYBRID'
            
            else:
                print(f"Warning: Unclear classification '{result}', defaulting to CONTEXTUAL")
                return "CONTEXTUAL"
        
        except Exception as e:
            print(f"Error during classification: {str(e)}, defaulting to CONTEXTUAL")
            return "CONTEXTUAL"
    
    def extract_sales_parameters(self, query):
        extraction_template = """Extract the following information from the sales query:

                                Query: {query}

                                Extract these fields. If a field is not mentioned or cannot be determined, write "NONE":
                                - store: Store name (e.g., "Bhat-Bhateni", "Kathmandu Mart")
                                - department: Department name (e.g., "Electronics", "Grocery")
                                - start_date: Specific start date ONLY if explicitly mentioned in YYYY-MM-DD format
                                - end_date: Specific end date ONLY if explicitly mentioned in YYYY-MM-DD format
                                - year: Specific year if mentioned (e.g., 2024, 2025)
                                - aggregation: Type of calculation (total, average, compare, trend)

                                IMPORTANT RULES:
                                - For start_date and end_date: ONLY extract if a specific date is given. Do NOT write explanations.
                                - If query mentions a holiday/festival name (like Dashain, Tihar) but no specific date, write "NONE" for dates.
                                - For holidays, we'll use contextual information separately.

                                Respond in this exact format (one word/phrase per line):
                                STORE: [store name or NONE]
                                DEPT: [department or NONE]
                                START_DATE: [YYYY-MM-DD or NONE]
                                END_DATE: [YYYY-MM-DD or NONE]
                                YEAR: [year number or NONE]
                                AGGREGATION: [type or total]

                                Response:"""    
        try:
            prompt = PromptTemplate.from_template(extraction_template)
            chain = prompt | self.llm |StrOutputParser()
            result = chain.invoke({"query": query})
            
            
            #parsing the result to extract values
            params = {
                "store": self.extract_value(result, "STORE"), 
                "department": self.extract_value(result, "DEPT"),
                "start_date": self.extract_value(result, "START_DATE"),
                "end_date": self.extract_value(result, "END_DATE"),
                "year": self.extract_value(result, "YEAR"),
                "aggregation": self.extract_value(result, "AGGREGATION") or "total"
            }
            
            
            params = self.validate_params(params)
            
            return params
        
        except Exception as e:
            print(f"Parameter extraction error: {str(e)}")
            return {
                "store": None,
                "department": None,
                "start_date": None,
                "end_date": None,
                "year": None,
                "aggregation": "total"
            }
        
    def extract_value(self,text,key):
        pattern = rf"{key}:\s*\[?([^\]\n]+)\]?"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            value = match.group(1).strip()
            value_lower = value.lower()
            
            if value_lower in ["unknown", "none", "null", "", "n/a", "not mentioned", "not specified"]:
                return None
            
            if '(' in value :
                value = value.split('(')[0].strip()
                if not value or value.lower() in ["unknown", "none", "null"]:
                    return None
            
            return value
        
        return None

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate dates
        for date_field in ['start_date', 'end_date']:
            if params.get(date_field):
                date_str = params[date_field]
                # Check if it's a valid date format (YYYY-MM-DD)
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                    params[date_field] = None
        
        # Validate year
        if params.get('year'):
            year_str = str(params['year'])
            # Extract year number if it contains extra text
            year_match = re.search(r'\b(20\d{2})\b', year_str)
            if year_match:
                params['year'] = year_match.group(1)
            elif not re.match(r'^20\d{2}$', year_str):
                params['year'] = None
        
        # Clean store and department names
        for field in ['store', 'department']:
            if params.get(field):
                # Remove quotes and extra whitespace
                params[field] = params[field].strip('"\'').strip()
                # If it's too long or contains weird characters, likely an error
                if len(params[field]) > 50 or '\n' in params[field]:
                    params[field] = None
        
        return params
        
    def route(self, query):
        classification = self.classify_query(query)
        routing_info = {
            'query' : query,
            'classification' : classification,
            'sales_params' : None
        }
        
        if classification in ['SALES', 'HYBRID']:
            routing_info['sales_params'] = self.extract_sales_parameters(query)
        
        return routing_info