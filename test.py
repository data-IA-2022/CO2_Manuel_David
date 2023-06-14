import pytest
from utils import get_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# test connection to database

@pytest.fixture
def test_engine():
    # Create an in-memory SQLite database for testing
    engine = get_engine(echo_arg=True)
    yield engine
    # Teardown: close the database connection
    engine.dispose()

def test_database_connection(test_engine):
    # Create a session
    Session = sessionmaker(bind=test_engine)
    session = Session()

    # Perform a database operation (e.g., query, insert, update)
    result = session.execute(text("SELECT 1"))

    # Assert the result
    assert result.scalar() == 1

    # Clean up: close the session
    session.close()

# test secret key vault