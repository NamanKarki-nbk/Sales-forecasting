import os
import pandas as pd
from src import SalesForecastingSystem
from src import RAGBuilder
from src import RAGQueryEngine
from src import SalesQueryEngine1
from src import QueryRouter
from src import HybridQueryEngine


def generate_all_forecasts(sales_data_path: str, forecast_weeks: int= 104):
    
    #initialize forecasting system
    
    forecast_system = SalesForecastingSystem(sales_data_path)
    stores = forecast_system.df['Store'].unique()
    departments = forecast_system.df['Dept'].unique()
    
    #create forecasts directory
    forecast_dir = "data/forecasts"
    os.makedirs(forecast_dir, exist_ok=True)
    
    consolidated_path = os.path.join(forecast_dir, 'all_consolidated_forecasts.csv')
    
    #check if  the consolidated file exists or not
    
    if os.path.exists(consolidated_path):
        print(f"Consolidated forecast file already exists: {consolidated_path}.")
        return consolidated_path
    
    print("\n Generating forecasts for all products")
    result = forecast_system.train_all_combinations(stores=stores, departments=departments, forecast_periods=forecast_weeks)
    
    
    #consolidate all forecasts into a single csv
    all_forecasts = []
    
    for store in stores:
        for dept in departments:
            try:
                forecast_df = forecast_system.get_forecast(store, dept, forecast_weeks)
                if forecast_df is not None and len(forecast_df)>0:
                    all_forecasts.append(forecast_df)
            except Exception as e:
                print(f"Error retrieving forecast for Store {store}, Dept {dept}: {e}")
    
    #combine forecasts
    
    if all_forecasts:
        consolidated_df = pd.concat(all_forecasts, ignore_index=True)
        consolidated_df.to_csv(consolidated_path, index=False)
        print(f"\n Consolidated forecast saved to {consolidated_path}")
    else:
        print("No forecasts were generated.")
        return None
    
    return consolidated_path


def setup_system(
    sales_data_path: str,
    holiday_path: str,
    forecast_weeks: int = 104,
    rebuild_vector_store: bool = False,
    regenerate_forecasts: bool = False  
):
    
    #paths 
    vector_store_path = "vector_store"
    forecast_dir = "data/forecasts"
    consolidated_forecast_path = os.path.join(forecast_dir, 'all_consolidated_forecasts.csv')
    
    if regenerate_forecasts or not os.path.exists(consolidated_forecast_path):
        forecast_csv_path = generate_all_forecasts(sales_data_path, forecast_weeks)
    else:
        print("Using existing consolidated forecast file.")
        forecast_csv_path = consolidated_forecast_path
    
    #rag_builder for holiday kb    
    index_file = os.path.join(vector_store_path, "index.faiss")
    pkl_file = os.path.join(vector_store_path, "index.pkl")
    
    if rebuild_vector_store or not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        rag_builder = RAGBuilder(
            csv_path= holiday_path,
            vectorstore_path = vector_store_path,
            chunk_size=500,
            chunk_overlap=100
        )
        rag_builder.build_vector_store()
    else:
        print("Using existing vector store.")
    
    #query engines
    
    #sales query engine
    sales_engine = SalesQueryEngine1(
        historical_csv=sales_data_path,
        forecast_csv=forecast_csv_path
    )
    
    #rag query engine
    rag_engine = RAGQueryEngine(
        vectorstore_path=vector_store_path,
        llm_model="mistral",
        top_k=3
    )
    
    #query router
    router = QueryRouter(
        llm_model='mistral'
    )
    
    #hybrid query engine
    hybrid_engine = HybridQueryEngine(
        sales_engine=sales_engine,
        rag_engine=rag_engine,
        router=router,
        llm_model="mistral"
    )
    
    return {
        'sales_engine': sales_engine,
        'rag_engine': rag_engine,
        'router': router,
        'hybrid_engine': hybrid_engine,
        'forecast_path': forecast_csv_path
    }


def run_example_queries(hybrid_engine: HybridQueryEngine):
    
    print("\n" + "="*60)
    print("EXAMPLE QUERIES")
    print("="*60)
    
    example_queries = [
        # Sales queries
        "What will be the total sales for Electronics at Bhat-Bhateni in 2025?",
        "Compare sales between 2024 and 2025 for Kathmandu Mart Electronics",
        "What's the average weekly sales for Grocery department at Bhat-Bhateni?",
        
        # Contextual queries
        "What are the major holidays in Nepal?",
        "When is Dashain celebrated and what is its significance?",
        "Tell me about Tihar festival",
        
        # Hybrid queries
        "How do Dashain holidays affect sales in the Electronics department?",
        "What's the sales trend during major festivals?",
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"EXAMPLE {i}/{len(example_queries)}")
        print(f"{'='*60}")
        
        result = hybrid_engine.query(query, verbose=True)
        print(f"\n💡 Final Answer:\n{result['answer']}")
        
        if i < len(example_queries):
            input("\nPress Enter to continue to next example...")

def main():
    #configure paths
    config = {
        'sales_data_path': "data/sales/synthetic_nepal_sales_with_more_holidays.csv",
        'holiday_path': "data\knowledge_base\downloadable_kb_csv.csv",
        'forecast_weeks': 104,  
        'rebuild_vectorstore': False,
        'regenerate_forecasts': False
    }
    
    #validating the paths
    
    if not os.path.exists(config['sales_data_path']):
        print(f"Sales data file not found: {config['sales_data_path']}")
        return
    
    if not os.path.exists(config['holiday_path']):
        print(f"Holiday knowledge base file not found: {config['holiday_path']}")
        return
    
    #setup system
    try:
        system = setup_system(
            sales_data_path=config['sales_data_path'],
            holiday_path=config['holiday_path'],
            forecast_weeks=config['forecast_weeks'],
            rebuild_vector_store=config['rebuild_vectorstore'],
            regenerate_forecasts=config['regenerate_forecasts']
        )
        
        hybrid_engine = system['hybrid_engine']
        # Menu loop
        while True:
            print("\n" + "="*60)
            print("MAIN MENU")
            print("="*60)
            print("1. Run example queries")
            print("2. Interactive query mode")
            print("3. Single query")
            print("4. Exit")
            print("="*60)
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                run_example_queries(hybrid_engine)
            
            elif choice == "2":
                hybrid_engine.interactive_mode()
            
            elif choice == "3":
                query = input("\n💬 Enter your question: ").strip()
                if query:
                    result = hybrid_engine.query(query, verbose=True)
                    print(f"\n💡 Answer:\n{result['answer']}")
                else:
                    print("❌ Empty query")
            
            elif choice == "4":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-4.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()