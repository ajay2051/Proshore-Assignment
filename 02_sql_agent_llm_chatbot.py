import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST")
DB_NAME = os.environ.get("DB_NAME")

# Create database connection URI
connection_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
db = SQLDatabase.from_uri(connection_uri)

# LLM configuration
LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,  # Slightly higher temperature for more creative responses
    "max_tokens": 2000
}
llm = ChatOpenAI(**LLM_CONFIG)

# Define the database schema information
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
    
2. **user**: 
- Columns 
    id (Primary Key, AutoField)
    first_name (CharField, max length: 50, nullable, blank)
    last_name (CharField, max length: 50, nullable, blank)
    email (EmailField, unique)
    phone_number (CharField, max length: 20, nullable, blank)
    birth_date (DateField, nullable, blank)
    address (CharField, max length: 150, nullable, blank)
    role (CharField, choices, default value: GENERAL)
    is_staff (BooleanField, default value: False)
    is_superuser (BooleanField, default value: False)
    is_active (BooleanField, default value: True)
    user_organization_id (Foreign Key → organization.id, nullable, blank, on delete: SET_NULL)
    membership (CharField, choices, default value: BASIC)
    tokens (PositiveIntegerField, default value: 0)

"""

# Define the system role for the AI
AGENT_LLM_SYSTEM_ROLE = f"""
You are a friendly and knowledgeable AI assistant. Your task is to help users query a database and provide clear, concise, and conversational answers based on the results.

Here is the schema of the database:
{DATABASE_SCHEMA}

Rules:
1. Always provide a human-readable response.
2. If the query result is a number or a list, explain it in a conversational way.
3. If the query result is empty, let the user know politely.
4. If there's an error, explain it in simple terms and suggest what the user can do next.

Example:
- User: "How many products are there?"
- AI: "There are 150 products in the warehouse. Let me know if you'd like more details!"
"""

# Initialize tools and chains

execute_query = QuerySQLDatabaseTool(db=db) # Tool to execute SQL queries

# Define the prompt for SQL generation
sql_prompt = PromptTemplate.from_template(
    """
    You are a SQL expert. Given the following database schema:

    {schema}

    Generate a valid SQL query to answer the following question:
    {question}

    Return only the SQL query, nothing else.
    """
)

# Chain to generate SQL queries
write_query = (
    RunnablePassthrough.assign(schema=lambda x: DATABASE_SCHEMA)
    | sql_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().strip("```sql").strip("```"))  # Strip markdown and extra whitespace
)

# Define the answer prompt
answer_prompt = PromptTemplate.from_template(
    """
    You are a friendly AI assistant. Below is the result of a database query:

    Query Result: {result}

    Based on this result, provide a clear and conversational answer to the user's question: {question}
    """
)

# Define the answer chain
answer = answer_prompt | llm | StrOutputParser()

# Define the full chain
chain = (
    RunnablePassthrough.assign(query=write_query)  # Generate SQL query
    .assign(result=itemgetter("query") | execute_query)  # Execute SQL query
    | answer  # Provide a conversational answer
)

# Function to interact with the agent
def ask_agent(question: str) -> str:
    try:
        # Generate the SQL query
        query = write_query.invoke({"question": question}).strip()
        print(f"Generated Query: {query}")  # Debugging: log the query

        # Execute the query
        result = execute_query.invoke(query)
        return answer.invoke({"question": question, "result": result})
    except Exception as e:
        return f"Query error: {str(e)}. Please verify your question and database schema."


# Test the agent
if __name__ == "__main__":
    # Example questions
    test_questions = [
        "how many questions are there",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = ask_agent(question)
        print(f"Response: {response}")