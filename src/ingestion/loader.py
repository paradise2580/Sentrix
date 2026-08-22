"""
src/ingestion/loader.py

Role
----
The single door through which all structured data enters and leaves
MySQL. Nothing else in the codebase should call pandas.to_sql or
pandas.read_sql directly — everything routes through DataLoader so
connection handling, logging, and error wrapping stay in one place.
"""

import pandas as pd
from sqlalchemy import text

from src.ingestion.db import get_engine
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


class DataLoader:
    """Reads and writes SENTRIX tables between pandas and MySQL."""

    def __init__(self):
        self.engine = get_engine()

    def write_df(self, df: pd.DataFrame, table: str, if_exists: str = "append",
                 chunksize: int = 5000) -> None:
        """Write a DataFrame into a MySQL table."""
        try:
            # NOTE: method="multi" builds one giant multi-VALUES statement per
            # chunk, which blows past MySQL's max_allowed_packet on wide/large
            # tables. Default (row-wise executemany) is slower but safe at any size.
            df.to_sql(
                table, self.engine, if_exists=if_exists,
                index=False, chunksize=chunksize,
            )
            logger.info(f"Wrote {len(df)} rows to '{table}' (if_exists={if_exists})")
        except Exception as e:
            raise SentrixException(e, sys)

    def read_table(self, table: str, limit: int | None = None) -> pd.DataFrame:
        """Read an entire table (or a limited slice) back into a DataFrame."""
        try:
            query = f"SELECT * FROM {table}"
            if limit:
                query += f" LIMIT {limit}"
            df = pd.read_sql(query, self.engine)
            logger.info(f"Read {len(df)} rows from '{table}'")
            return df
        except Exception as e:
            raise SentrixException(e, sys)

    def read_query(self, sql: str) -> pd.DataFrame:
        """Run an arbitrary SELECT query and return the result as a DataFrame."""
        try:
            df = pd.read_sql(sql, self.engine)
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            raise SentrixException(e, sys)

    def truncate_table(self, table: str) -> None:
        """Empty a table without dropping its schema — used before a full reload."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
                conn.execute(text(f"TRUNCATE TABLE {table}"))
                conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
            logger.info(f"Truncated table '{table}'")
        except Exception as e:
            raise SentrixException(e, sys)

    def csv_to_mysql(self, csv_path: str, table: str, if_exists: str = "replace") -> None:
        """Load a CSV file straight into a MySQL table."""
        try:
            df = pd.read_csv(csv_path)
            self.write_df(df, table, if_exists=if_exists)
        except Exception as e:
            raise SentrixException(e, sys)


if __name__ == "__main__":
    loader = DataLoader()
    print("DataLoader initialised. Example table read:")
    print(loader.read_table("suppliers", limit=5))
