import pytest
from utils import get_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import joblib
import dotenv
from sklearn.multioutput import MultiOutputRegressor

# test connection to database

@pytest.fixture
def test_engine():
    # en local
    dotenv.load_dotenv()
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

def test_pikle_file():
    model = joblib.load('model.pkl')
    assert isinstance(model, MultiOutputRegressor)