"""
Helper functions for Snowflake operations
"""

import os
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Try to import Snowflake libraries
try:
    import snowflake.connector
    SNOWFLAKE_CONNECTOR_AVAILABLE = True
except ImportError:
    SNOWFLAKE_CONNECTOR_AVAILABLE = False
    logger.warning("snowflake-connector-python not available")

try:
    from snowflake.snowpark import Session
    SNOWPARK_AVAILABLE = True
except ImportError:
    SNOWPARK_AVAILABLE = False
    logger.warning("snowflake-snowpark-python not available")


def get_snowflake_session():
    """
    Get Snowflake session from secrets.toml or environment variables
    
    Returns:
        Snowflake session (Snowpark Session or connector connection) or None
    """
    config = {}
    config_loaded = False
    
    # Try to load from various locations (in order of preference)
    secrets_paths = [
        # Package-specific config.toml (highest priority for this package)
        Path(__file__).parent / "config.toml",
        Path.cwd() / "config.toml",
        # Standard .streamlit/secrets.toml locations
        Path(".streamlit/secrets.toml"),
        Path(__file__).parent.parent / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    
    for secrets_path in secrets_paths:
        if secrets_path.exists():
            try:
                # Try using tomllib (Python 3.11+)
                try:
                    import tomllib
                    with open(secrets_path, 'rb') as f:
                        secrets = tomllib.load(f)
                        # Check both [connections.snowflake] and direct [snowflake] sections
                        if "connections" in secrets and "snowflake" in secrets["connections"]:
                            config = secrets["connections"]["snowflake"]
                            config_loaded = True
                        elif "snowflake" in secrets:
                            # Direct [snowflake] section (for config.toml)
                            config = secrets["snowflake"]
                            config_loaded = True
                except ImportError:
                    # Fallback to toml library
                    try:
                        import toml
                        with open(secrets_path, 'r') as f:
                            secrets = toml.load(f)
                            # Check both [connections.snowflake] and direct [snowflake] sections
                            if "connections" in secrets and "snowflake" in secrets["connections"]:
                                config = secrets["connections"]["snowflake"]
                                config_loaded = True
                            elif "snowflake" in secrets:
                                # Direct [snowflake] section (for config.toml)
                                config = secrets["snowflake"]
                                config_loaded = True
                    except ImportError:
                        pass
                except Exception:
                    pass
                
                # Fallback: simple TOML parsing
                if not config_loaded:
                    try:
                        with open(secrets_path, 'r') as f:
                            content = f.read()
                            import re
                            # Try [connections.snowflake] first
                            snowflake_section = re.search(r'\[connections\.snowflake\](.*?)(?=\[|$)', content, re.DOTALL)
                            if not snowflake_section:
                                # Try direct [snowflake] section (for config.toml)
                                snowflake_section = re.search(r'\[snowflake\](.*?)(?=\[|$)', content, re.DOTALL)
                            
                            if snowflake_section:
                                section_content = snowflake_section.group(1)
                                for line in section_content.split('\n'):
                                    line = line.strip()
                                    if '=' in line and not line.startswith('#'):
                                        key, value = line.split('=', 1)
                                        key = key.strip()
                                        value = value.strip().strip('"').strip("'")
                                        config[key] = value
                                        config_loaded = True
                    except Exception:
                        pass
            except Exception:
                pass
            
            if config_loaded:
                break
    
    # Fall back to environment variables
    if not config:
        config = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "SNOWFLAKE_SAMPLE_DATA"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "TPCH_SF1"),
        }
    
    # Validate config
    if not config or not all([config.get("account"), config.get("user"), config.get("password")]):
        return None
    
    # Try Snowpark first
    if SNOWPARK_AVAILABLE:
        try:
            session = Session.builder.configs(config).create()
            session.sql("SELECT CURRENT_USER()").collect()
            return session
        except Exception:
            pass
    
    # Fall back to connector
    if SNOWFLAKE_CONNECTOR_AVAILABLE:
        try:
            conn = snowflake.connector.connect(
                account=config["account"],
                user=config["user"],
                password=config["password"],
                warehouse=config.get("warehouse", "COMPUTE_WH"),
                database=config.get("database", "SNOWFLAKE_SAMPLE_DATA"),
                schema=config.get("schema", "TPCH_SF1"),
            )
            cursor = conn.cursor()
            cursor.execute("SELECT CURRENT_USER()")
            cursor.fetchone()
            cursor.close()
            return conn
        except Exception:
            pass
    
    return None


def get_table_columns(session, table_name: str, schema_name: str) -> List[str]:
    """
    Get column names from a table
    
    Args:
        session: Snowflake session (Snowpark or connector)
        table_name: Name of the table
        schema_name: Full schema name (e.g., 'SNOWFLAKE_SAMPLE_DATA.TPCH_SF1')
    
    Returns:
        List of column names, empty list on error
    """
    try:
        if not table_name or not schema_name:
            raise ValueError(f"Invalid table_name or schema_name: table={table_name}, schema={schema_name}")
        
        desc_query = f"DESCRIBE TABLE {schema_name}.{table_name}"
        logger.debug(f"Executing: {desc_query}")
        
        if hasattr(session, 'sql'):  # Snowpark
            desc_df = session.sql(desc_query).to_pandas()
        else:  # Connector
            cursor = session.cursor()
            try:
                cursor.execute(desc_query)
                columns = cursor.fetchall()
                import pandas as pd
                desc_df = pd.DataFrame(
                    columns, 
                    columns=[desc[0] for desc in cursor.description] if cursor.description else ['name', 'type']
                )
            finally:
                cursor.close()
        
        if desc_df.empty:
            logger.warning(f"DESCRIBE returned empty result for {schema_name}.{table_name}")
            return []
        
        # Find column name column
        desc_name_col = None
        for col_name in ['name', 'NAME', 'column_name', 'COLUMN_NAME', 'Name', 'COLUMN']:
            if col_name in desc_df.columns:
                desc_name_col = col_name
                break
        
        if desc_name_col is None:
            for col in desc_df.columns:
                if 'name' in str(col).lower():
                    desc_name_col = col
                    break
        
        if desc_name_col is None and len(desc_df.columns) > 0:
            desc_name_col = desc_df.columns[0]
        
        # Get column names
        if desc_name_col and desc_name_col in desc_df.columns:
            column_names = desc_df[desc_name_col].tolist()
        else:
            column_names = desc_df.iloc[:, 0].tolist() if len(desc_df.columns) > 0 else []
        
        # Filter out None/empty values
        column_names = [col for col in column_names if col]
        
        logger.debug(f"Found {len(column_names)} columns for {schema_name}.{table_name}")
        return column_names
        
    except Exception as e:
        logger.error(
            f"Error getting columns for {schema_name}.{table_name}: {e}",
            exc_info=True
        )
        return []
