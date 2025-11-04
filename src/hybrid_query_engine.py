from typing import Dict, Any, Optional
from src import  SalesQueryEngine1, RAGQueryEngine, QueryRouter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime, timedelta

class HybridQueryEngine:
    
    def __init__( 
        self,
        sales_engine: SalesQueryEngine1,
        rag_engine: RAGQueryEngine,
        router: Optional[QueryRouter] = None,
        llm_model: str = "mistral"
    ):
        
        self.sales_engine = sales_engine
        self.rag_engine = rag_engine
        self.router = router or QueryRouter(llm_model=llm_model)
        self.llm = OllamaLLM(model=llm_model, temperature=0.2)
        
        print("\n" + "="*60)
        print("HYBRID QUERY ENGINE INITIALIZED")
        print("="*60)
        print("✓ Sales Query Engine: Ready")
        print("✓ RAG Query Engine: Ready")
        print("✓ Query Router: Ready")
        print("="*60)
    
    def query(self, user_query: str, verbose: bool = True) -> Dict[str, Any]:
        if verbose:
            print(f"\n🔍 Query: {user_query}")
            print("-" * 60)
        
        # Step 1: Route the query
        routing = self.router.route(user_query)
        classification = routing['classification']
        
        if verbose:
            print(f"📊 Classification: {classification}")
        
        # Step 2: Process based on classification
        if classification == "SALES":
            return self._handle_sales_query(user_query, routing, verbose)
        
        elif classification == "CONTEXTUAL":
            return self._handle_contextual_query(user_query, verbose)
        
        elif classification == "HYBRID":
            return self._handle_hybrid_query(user_query, routing, verbose)
        
        else:
            return {
                "answer": "I'm not sure how to process this query.",
                "classification": "UNKNOWN",
                "error": "Unknown classification"
            }
    
    def _handle_sales_query(
    self, 
    query: str, 
    routing: Dict[str, Any], 
    verbose: bool
) -> Dict[str, Any]:
        """Handle pure sales queries with enhanced capabilities"""
        
        params = routing.get('sales_params', {})
        
        if verbose:
            print(f"🔢 Processing as SALES query")
            print(f"   Extracted params: {params}")
        
        try:
            # Extract parameters
            store = params.get('store')
            dept = params.get('department')
            aggregation = params.get('aggregation', 'total')
            start_date = params.get('start_date')
            end_date = params.get('end_date')
            year = params.get('year')
            
            # Handle missing store
            if not store or store == "unknown":
                available_stores = self.sales_engine.stores
                return {
                    "answer": f"I need to know which store you're asking about. Available stores: {', '.join(available_stores)}",
                    "classification": "SALES",
                    "missing_params": ["store"],
                    "suggestions": available_stores
                }
            
            # Handle missing department
            if not dept or dept == "unknown":
                available_depts = self.sales_engine.departments
                return {
                    "answer": f"I need to know which department at {store} you're asking about. Available departments: {', '.join(available_depts[:10])}{'...' if len(available_depts) > 10 else ''}",
                    "classification": "SALES",
                    "missing_params": ["department"],
                    "suggestions": available_depts
                }
            
            query_lower = query.lower()
            
            # Detect month first (to set date range)
            months = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            
            detected_month = None
            detected_month_name = None
            for month_name, month_num in months.items():
                if month_name in query_lower:
                    detected_month = month_num
                    detected_month_name = month_name
                    break
            
            # Set date range if month is detected
            if detected_month:
                from datetime import datetime
                year_int = int(year) if year else 2025
                
                # Calculate start and end dates for the month
                start_date = f"{year_int}-{detected_month:02d}-01"
                
                # Calculate last day of month
                if detected_month == 12:
                    end_date = f"{year_int}-12-31"
                else:
                    next_month = detected_month + 1
                    last_day = (datetime(year_int, next_month, 1) - timedelta(days=1)).day
                    end_date = f"{year_int}-{detected_month:02d}-{last_day}"
            
            # NOW check what type of analysis is needed
            # Check for holiday impact query
            if 'holiday' in query_lower or 'festival' in query_lower:
                year_int = int(year) if year else 2025
                result = self.sales_engine.get_holiday_impact(store, dept, year_int)
                
                # Also get the total sales for the detected month if applicable
                total_sales = None
                if detected_month:
                    total_sales = self.sales_engine.total_sales(
                        store=store,
                        dept=dept,
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if result and 'message' not in result:
                    uplift = result.get('uplift_percentage', 0)
                    answer = ""
                    
                    # Add month-specific sales if detected
                    if detected_month and total_sales:
                        answer += f"Total sales for {dept} at {store} in {detected_month_name.capitalize()} {year_int}: Rs. {total_sales:,.2f}\n\n"
                    
                    answer += f"Holiday Impact Analysis for {dept} at {store} in {year_int}:\n\n"
                    answer += f"Holiday Period Sales:\n"
                    answer += f"  - Total: Rs. {result['holiday_total']:,.2f}\n"
                    answer += f"  - Average per week: Rs. {result['holiday_avg']:,.2f}\n"
                    answer += f"  - Number of weeks: {result['holiday_weeks']}\n\n"
                    answer += f"Non-Holiday Period Sales:\n"
                    answer += f"  - Total: Rs. {result['non_holiday_total']:,.2f}\n"
                    answer += f"  - Average per week: Rs. {result['non_holiday_avg']:,.2f}\n"
                    answer += f"  - Number of weeks: {result['non_holiday_weeks']}\n\n"
                    
                    if uplift:
                        answer += f"Holiday Uplift: {uplift:+.1f}%\n"
                        if uplift > 50:
                            answer += "Holidays have a STRONG positive impact on sales."
                        elif uplift > 20:
                            answer += "Holidays have a MODERATE positive impact on sales."
                        else:
                            answer += "Holidays have a SLIGHT positive impact on sales."
                    
                    return {
                        "answer": answer,
                        "classification": "SALES",
                        "params": params,
                        "result": {**result, 'month_total': total_sales} if detected_month else result,
                        "success": True
                    }
            
            # Trend query
            elif 'trend' in query_lower:
                result = self.sales_engine.get_sales_trend(
                    store, dept, start_date, end_date, period='monthly'
                )
                
                if result:
                    answer = f"Sales Trend for {dept} at {store}:\n\n"
                    for period_data in result[-6:]:  # Last 6 periods
                        answer += f"{period_data['Period']}: Rs. {period_data['Total_Sales']:,.2f} "
                        answer += f"(Avg: Rs. {period_data['Avg_Sales']:,.2f})\n"
                    
                    return {
                        "answer": answer,
                        "classification": "SALES",
                        "params": params,
                        "result": result,
                        "success": True
                    }
            
            # Summary query
            elif 'summary' in query_lower or 'overview' in query_lower:
                result = self.sales_engine.get_sales_summary(store, dept, start_date, end_date)
                
                answer = f"Sales Summary for {dept} at {store}:\n\n"
                answer += f"Total Sales: Rs. {result['total_sales']:,.2f}\n"
                answer += f"Average Weekly: Rs. {result['average_weekly_sales']:,.2f}\n"
                answer += f"Range: Rs. {result['min_weekly_sales']:,.2f} - Rs. {result['max_weekly_sales']:,.2f}\n"
                answer += f"Total Weeks: {result['total_weeks']} ({result['historical_weeks']} historical, {result['forecast_weeks']} forecast)\n"
                answer += f"Period: {result['date_range']['start']} to {result['date_range']['end']}"
                
                return {
                    "answer": answer,
                    "classification": "SALES",
                    "params": params,
                    "result": result,
                    "success": True
                }
            
            # Comparison query
            elif aggregation and "compare" in aggregation.lower():
                year_int = int(year) if year else 2025
                result = self.sales_engine.compare_sales(
                    store=store,
                    dept=dept,
                    year1=year_int - 1,
                    year2=year_int
                )
                
                answer = self._format_comparison_answer(store, dept, result)
                
                return {
                    "answer": answer,
                    "classification": "SALES",
                    "params": params,
                    "result": result,
                    "success": True
                }
            
            # Average query
            elif aggregation and "average" in aggregation.lower():
                avg_sales = self.sales_engine.average_weekly_sales(
                    store=store,
                    dept=dept,
                    start_date=start_date,
                    end_date=end_date
                )
                
                period_str = ""
                if start_date and end_date:
                    period_str = f" between {start_date} and {end_date}"
                elif year:
                    period_str = f" in {year}"
                
                answer = f"The average weekly sales for {dept} at {store}{period_str} is Rs. {avg_sales:,.2f}"
                
                return {
                    "answer": answer,
                    "classification": "SALES",
                    "params": params,
                    "success": True
                }
            
            # Default: Total sales (including month-specific if detected)
            else:
                total_sales = self.sales_engine.total_sales(
                    store=store,
                    dept=dept,
                    start_date=start_date,
                    end_date=end_date
                )
                
                period_str = ""
                if detected_month:
                    period_str = f" in {detected_month_name.capitalize()} {year}"
                elif start_date and end_date:
                    period_str = f" from {start_date} to {end_date}"
                elif year:
                    period_str = f" in {year}"
                else:
                    period_str = " (all available data)"
                
                answer = f"The total sales for {dept} at {store}{period_str} is Rs. {total_sales:,.2f}"
                
                return {
                    "answer": answer,
                    "classification": "SALES",
                    "params": params,
                    "success": True
                }
        
        except Exception as e:
            error_msg = f"Error processing sales query: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
            
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"I encountered an error while processing your sales query: {str(e)}",
                "classification": "SALES",
                "error": str(e),
                "success": False
            }
    
    
    def _handle_contextual_query(self, query: str, verbose: bool) -> Dict[str, Any]:
        """Handle contextual/RAG queries"""
        
        if verbose:
            print(f"📚 Processing as CONTEXTUAL query")
        
        try:
            # Get answer from RAG system
            answer = self.rag_engine.ask(query)
            
            if verbose:
                print(f"✅ Answer generated from knowledge base")
            
            return {
                "answer": answer,
                "classification": "CONTEXTUAL",
                "success": True
            }
        
        except Exception as e:
            error_msg = f"Error processing contextual query: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
            
            return {
                "answer": f"I encountered an error while searching the knowledge base: {str(e)}",
                "classification": "CONTEXTUAL",
                "error": str(e),
                "success": False
            }
    
    def _handle_hybrid_query(
    self, 
    query: str, 
    routing: Dict[str, Any], 
    verbose: bool
) -> Dict[str, Any]:
        """Handle queries needing both sales data and context"""
        
        if verbose:
            print(f"🔀 Processing as HYBRID query")
        
        try:
            # Check if sales_params exist, if not, re-route as pure SALES
            if 'sales_params' not in routing or not routing['sales_params']:
                if verbose:
                    print(f"⚠️ No sales params found, treating as pure SALES query")
                return self._handle_sales_query(query, {'sales_params': {}}, verbose)
            
            # Get sales data
            sales_result = self._handle_sales_query(query, routing, verbose=False)
            
            # Check if sales query was successful
            if not sales_result or not sales_result.get('success', False):
                # If sales query failed, try to get contextual info only
                if verbose:
                    print(f"⚠️ Sales query failed, attempting contextual only")
                
                sales_answer = sales_result.get('answer', 'Unable to retrieve sales data') if sales_result else 'Unable to retrieve sales data'
                
                # Still try to get contextual information
                try:
                    contextual_result = self._handle_contextual_query(query, verbose=False)
                    contextual_answer = contextual_result.get('answer', 'No contextual information available')
                    
                    # Return combined answer even if sales failed
                    combined_answer = f"{sales_answer}\n\nAdditional Context:\n{contextual_answer}"
                    
                    return {
                        "answer": combined_answer,
                        "classification": "HYBRID",
                        "sales_component": sales_answer,
                        "contextual_component": contextual_answer,
                        "success": False,
                        "partial": True
                    }
                except Exception as e:
                    # Both failed
                    return {
                        "answer": f"I encountered an error processing your query. {sales_answer}",
                        "classification": "HYBRID",
                        "error": str(e),
                        "success": False
                    }
            
            sales_answer = sales_result.get('answer', 'No sales data available')
            
            # Get contextual information
            contextual_result = self._handle_contextual_query(query, verbose=False)
            contextual_answer = contextual_result.get('answer', 'No contextual information available')
            
            # Synthesize combined answer
            synthesis_template = """You are a business intelligence assistant. Combine the following information to answer the user's question comprehensively.

    User Question: {query}

    Sales Data: {sales_data}

    Contextual Information: {context}

    Provide a clear, comprehensive answer that integrates both the numerical sales data and the contextual information. Be concise but complete.

    Answer:"""
            
            prompt = PromptTemplate.from_template(synthesis_template)
            chain = prompt | self.llm | StrOutputParser()
            
            final_answer = chain.invoke({
                "query": query,
                "sales_data": sales_answer,
                "context": contextual_answer
            })
            
            if verbose:
                print(f"✅ Hybrid answer synthesized")
            
            return {
                "answer": final_answer,
                "classification": "HYBRID",
                "sales_component": sales_answer,
                "contextual_component": contextual_answer,
                "success": True
            }
        
        except Exception as e:
            error_msg = f"Error processing hybrid query: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
            
            import traceback
            traceback.print_exc()
            
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "classification": "HYBRID",
                "error": str(e),
                "success": False
            }
        
    def _format_comparison_answer(
        self, 
        store: str, 
        dept: str, 
        result: Dict[str, Any]
    ) -> str:
        
        year1_total = result['year1_total']
        year2_total = result['year2_total']
        difference = result['difference']
        pct_change = result.get('pct_change')
        
        direction = "increased" if difference > 0 else "decreased"
        
        answer = f"For {dept} at {store}:\n\n"
        answer += f"Previous year: Rs. {year1_total:,.2f}\n"
        answer += f"Current year: Rs. {year2_total:,.2f}\n"
        answer += f"Difference: Rs. {abs(difference):,.2f}\n"
        
        if pct_change is not None:
            answer += f"Change: {abs(pct_change):.2f}% {direction}"
        
        return answer
    
    def interactive_mode(self):

        
        print("\n" + "="*60)
        print("HYBRID QUERY SYSTEM - INTERACTIVE MODE")
        print("="*60)
        print("Ask questions about sales forecasts and business context.")
        print("Type 'quit' or 'exit' to stop.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n💬 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Process query
                result = self.query(user_input, verbose=True)
                
                print(f"\n💡 Answer: {result['answer']}")
                print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

