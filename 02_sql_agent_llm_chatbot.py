
import logging
import os
from operator import itemgetter

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_db_connection():
    """Safely retrieve database connection parameters"""
    try:
        DB_USER = os.getenv("DB_USER")
        DB_PASSWORD = os.getenv("DB_PASS")
        DB_HOST = os.getenv("DB_HOST")
        DB_NAME = os.getenv("DB_NAME")

        if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
            raise ValueError("Missing database configuration environment variables")

        connection_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
        return SQLDatabase.from_uri(connection_uri)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


def initialize_llm():
    """Initialize and validate LLM configuration"""
    try:
        LLM_CONFIG = {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 2000
        }

        if not LLM_CONFIG["api_key"]:
            raise ValueError("Missing OpenAI API key")

        return ChatOpenAI(**LLM_CONFIG)
    except Exception as e:
        logger.error(f"LLM initialization error: {e}")
        raise


DATABASE_SCHEMA = """
Here is the schema of the database:

1. **question**: Question
- Columns
    id (Primary Key, AutoField)
    title (CharField, nullable, blank)
    eval_skill (CharField, choices, default value: GENERAL_CONTENT)
    qn_num (IntegerField, nullable, blank)
    qn_type (CharField, choices, default value: SINGLE_CHOICE)
    score (IntegerField, nullable, blank)
    status (CharField, choices, default value: CREATED)
    head (CharField, unique, max length: 1000)
    statement (TextField, nullable, blank)
    question_content_text (TextField, nullable, blank)
    answer_content_is_char_limited (BooleanField, default value: False)
    answer_content_model_answer (TextField, nullable, blank)
    answer_options_text (TextField, nullable, blank)
    answer_explanation_text (TextField, nullable, blank)
    assigned_at (DateTimeField, nullable, blank)
    reviewed_at (DateTimeField, nullable, blank)
    reason (TextField, nullable, blank)
    owner_id (Foreign Key → user_auth_authuser.id, nullable, blank)
    reviewers_id (Foreign Key → user_auth_authuser.id, nullable, blank)
    category_id (Foreign Key → question_category.id, nullable)
    question_group_id (Foreign Key → question_group.id, nullable, blank)
"""

AGENT_LLM_SYSTEM_ROLE = f"""
You are a friendly and knowledgeable AI assistant. 
Your task is to help users query a database and provide clear, concise, and conversational answers based on the results."""

def create_query_chain(db, llm):
    """Create the query processing chain with error handling"""
    try:
        execute_query = QuerySQLDatabaseTool(db=db)

        sql_prompt = PromptTemplate.from_template(
            """
            You are a SQL expert. Given the following database schema:

            {schema}

            Generate a valid SQL query to answer the following question:
            {question}

            Return only the SQL query, nothing else.
            """
        )

        write_query = (
                RunnablePassthrough.assign(schema=lambda x: DATABASE_SCHEMA)
                | sql_prompt
                | llm
                | StrOutputParser()
                | (lambda x: x.strip().strip("```sql").strip("```"))
        )

        answer_prompt = PromptTemplate.from_template(
            """
            You are a friendly AI assistant. Below is the result of a database query:

            Query Result: {result}

            Based on this result, provide a clear and conversational answer to the user's question: {question}
            """
        )

        answer = answer_prompt | llm | StrOutputParser()

        return RunnablePassthrough.assign(query=write_query).assign(result=itemgetter("query") | execute_query) | answer
    except Exception as e:
        logger.error(f"Chain creation error: {e}")
        raise


def ask_agent(question: str) -> str:
    """Main function to process user questions with comprehensive error handling"""
    try:
        db = get_db_connection()
        llm = initialize_llm()

        chain = create_query_chain(db, llm)

        result = chain.invoke({"question": question})
        logger.info(f"Successfully processed query for: {question}")
        return result
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        return f"Configuration error: {ve}. Please check your environment setup."
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        return f"Sorry, an unexpected error occurred: {e}. Please try again or contact support."


if __name__ == "__main__":
    try:
        test_question = "How many questions are in the database?"
        response = ask_agent(test_question)
        print(response)
    except Exception as e:
        print(f"Test execution failed: {e}")
