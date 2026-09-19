import sqlite3
from sqlite3 import Error



def get_connections():
    "Database connection logic here."
    try:
        conn = sqlite3.connect("database.db")
        print("Database connected successfully.") 
        return conn

    except sqlite3.IntegrityError as e:
        raise sqlite3.IntegrityError(
            f"Failed to save message: {e}"
        ) from e
    
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError (
        f"Couldn't perform the operation {e}"
        ) from e

    except sqlite3.ProgrammingError as e:
        raise sqlite3.ProgrammingError(
            f"Invalid api {e}"
        ) from e

        
def create_table(conn):
    "Creating the sessions table."

    try:
        c = conn.cursor()
        sql_create_table = '''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        );
        '''
        c.execute(sql_create_table)
        conn.commit()

    except sqlite3.IntegrityError as e:
        raise sqlite3.IntegrityError(
            f"Failed to save message: {e}"
        ) from e
    
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError (
        f"Couldn't perform the operation {e}"
        ) from e

    except sqlite3.ProgrammingError as e:
        raise sqlite3.ProgrammingError(
            f"Invalid api {e}"
        ) from e


def save_message(
    conn,
    session_id: str,
    role: str,
    content: str,
):
    "Saving message in the database"
    try:
        c = conn.cursor()
        sql = """
        INSERT INTO messages(session_id, role, content) VALUES (?,?,?);
        """
        c.execute(sql, (session_id, role, content))
        conn.commit()
        print(f"Message is saved {session_id},{role}")

    except sqlite3.IntegrityError as e:
        raise sqlite3.IntegrityError(
            f"Failed to save message: {e}"
        ) from e
    
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError (
        f"Couldn't perform the operation {e}"
        ) from e

    except sqlite3.ProgrammingError as e:
        raise sqlite3.ProgrammingError(
            f"Invalid api {e}"
        ) from e

def get_recent_messages(
    conn,
    session_id: str,
    limit: int = 10,
):

    "Retriving the recent messages."
    try:
        c = conn.cursor()
        sql_query = f"""
        SELECT * from messages
        WHERE session_id = {session_id}
        ORDER BY id DESC
        LIMIT {limit};
        """
        c.execute(sql_query)
        rows = c.fetchall()

        return rows

    except sqlite3.IntegrityError as e:
        raise sqlite3.IntegrityError(
            f"Failed to save message: {e}"
        ) from e
    
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError (
        f"Couldn't perform the operation {e}"
        ) from e

    except sqlite3.ProgrammingError as e:
        raise sqlite3.ProgrammingError(
            f"Invalid api {e}"
        ) from e
