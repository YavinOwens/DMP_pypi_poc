#!/usr/bin/env python3
"""
Data Quality Rules Manager
Manages CSV file for validation rules with LLM-powered analysis and updates
"""

import csv
import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Configure logging - use smart configuration
try:
    from ..logging_config import configure_logging
    # Only configure if not already configured
    if not logging.getLogger().handlers:
        configure_logging(verbose=False)
except ImportError:
    # Fallback - set to WARNING to reduce verbosity
    logging.basicConfig(level=logging.WARNING)

# Optional imports - handle ImportError if modules not available
# Note: RAGSystem and execute_cortex_query are imported lazily to avoid circular imports
RAGSystem = None
execute_cortex_query = None

def _lazy_import_rag_system():
    """Lazy import RAGSystem to avoid circular imports"""
    global RAGSystem
    if RAGSystem is None:
        try:
            from ..rag.system import RAGSystem as _RAGSystem
            RAGSystem = _RAGSystem
        except ImportError:
            pass
    return RAGSystem

def _lazy_import_execute_cortex_query():
    """Lazy import execute_cortex_query to avoid circular imports"""
    global execute_cortex_query
    if execute_cortex_query is None:
        try:
            from ..benchmark import execute_cortex_query as _execute_cortex_query
            execute_cortex_query = _execute_cortex_query
        except ImportError:
            pass
    return execute_cortex_query

logger = logging.getLogger(__name__)

# Data Quality Dimensions (standard DAMA DMBOK dimensions)
DATA_QUALITY_DIMENSIONS = [
    "Completeness",
    "Accuracy",
    "Consistency",
    "Timeliness",
    "Validity",
    "Uniqueness",
    "Integrity",
    "Conformity",
    "Precision",
    "Currency"
]

# RACI values
RACI_VALUES = ["Responsible", "Accountable", "Consulted", "Informed"]

# Status values
STATUS_VALUES = ["Draft", "Active", "Inactive", "Deprecated"]


class DataQualityRulesManager:
    """Manages data quality validation rules CSV file with LLM integration"""
    
    def __init__(self, csv_path: Optional[str] = None, rag_system=None):
        """
        Initialize DataQualityRulesManager
        
        Args:
            csv_path: Path to CSV file. If None, looks in package root, then current directory.
            rag_system: Optional RAGSystem instance for DAMA DMBOK references
        """
        if csv_path is None:
            # Try package root first, then current directory
            package_root = Path(__file__).parent.parent.parent
            csv_path = package_root / "data_quality_rules.csv"
            if not csv_path.exists():
                csv_path = Path.cwd() / "data_quality_rules.csv"
        else:
            csv_path = Path(csv_path)
        self.csv_path = csv_path
        self.rules: List[Dict[str, Any]] = []
        self.rag_system = rag_system
        self._ensure_csv_exists()
        self._load_rules()
    
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            logger.info(f"Creating new CSV file: {self.csv_path}")
            headers = [
                "validation_number",
                "business_rule",
                "data_quality_rule",
                "data_quality_dimension",
                "Code Snowflake",
                "Code SQL",
                "Code Python",
                "Dama Reference",
                "raci",
                "status",
                "created_date",
                "created_by",
                "updated_date",
                "updated_by",
                "version",
                "notes"
            ]
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
        else:
            logger.info(f"Using existing CSV file: {self.csv_path}")
            # Ensure existing CSV has all required columns
            self._ensure_all_columns_exist()
    
    def _ensure_all_columns_exist(self):
        """Ensure all required columns exist in the CSV, add missing ones"""
        if not self.csv_path.exists():
            return
        
        required_columns = [
            "validation_number",
            "business_rule",
            "data_quality_rule",
            "data_quality_dimension",
            "Code Snowflake",
            "Code SQL",
            "Code Python",
            "Dama Reference",
            "raci",
            "status",
            "created_date",
            "created_by",
            "updated_date",
            "updated_by",
            "version",
            "notes"
        ]
        
        try:
            # Read existing CSV
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_headers = reader.fieldnames or []
                rows = list(reader)
            
            # Check if any columns are missing
            missing_columns = [col for col in required_columns if col not in existing_headers]
            
            if missing_columns:
                logger.info(f"Adding missing columns to CSV: {missing_columns}")
                # Add missing columns to each row
                for row in rows:
                    for col in missing_columns:
                        row[col] = ""
                
                # Write back with all columns
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=required_columns)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            logger.warning(f"Could not ensure all columns exist: {e}")
    
    def _load_rules(self):
        """Load rules from CSV file"""
        self.rules = []
        if self.csv_path.exists():
            try:
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert empty strings to None for cleaner data
                        cleaned_row = {k: (v if v else None) for k, v in row.items()}
                        self.rules.append(cleaned_row)
                logger.info(f"Loaded {len(self.rules)} rules from CSV")
            except Exception as e:
                logger.error(f"Error loading rules from CSV: {e}")
                self.rules = []
    
    def _save_rules(self):
        """Save rules to CSV file"""
        if not self.rules:
            return
        
        headers = list(self.rules[0].keys())
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(self.rules)
            logger.info(f"Saved {len(self.rules)} rules to CSV")
        except Exception as e:
            logger.error(f"Error saving rules to CSV: {e}")
            raise
    
    def get_next_validation_number(self) -> str:
        """Generate next validation number (e.g., VR-001, VR-002)"""
        existing_numbers = [
            int(r.get('validation_number', '').split('-')[-1])
            for r in self.rules
            if r.get('validation_number') and r['validation_number'].startswith('VR-')
        ]
        next_num = max(existing_numbers, default=0) + 1
        return f"VR-{next_num:03d}"
    
    def add_rule(self, business_rule: str, data_quality_rule: str, 
                 data_quality_dimension: str, raci: str,
                 created_by: str = "System", status: str = "Draft",
                 notes: Optional[str] = None,
                 code_snowflake: Optional[str] = None,
                 code_sql: Optional[str] = None,
                 code_python: Optional[str] = None,
                 dama_reference: Optional[str] = None,
                 session: Optional[Any] = None,
                 auto_fetch_dama: bool = True) -> str:
        """
        Add a new validation rule
        
        Args:
            business_rule: Business requirement description
            data_quality_rule: Technical validation rule
            data_quality_dimension: Data quality dimension
            raci: RACI assignment
            created_by: User/system creating the rule
            status: Rule status (default: "Draft")
            notes: Additional notes
            code_snowflake: Snowflake code for validation (optional)
            code_sql: SQL code for validation (optional)
            code_python: Python code for validation (optional)
            dama_reference: DAMA DMBOK reference (optional, auto-fetched if auto_fetch_dama=True)
            session: Snowflake session for RAG (if auto_fetch_dama=True and rag_system not initialized)
            auto_fetch_dama: Whether to automatically fetch DAMA references using RAG (default: True)
        
        Returns:
            validation_number: The generated validation number
        """
        validation_number = self.get_next_validation_number()
        now = datetime.now().isoformat()
        
        # Auto-fetch DAMA reference if not provided and auto_fetch is enabled
        if not dama_reference and auto_fetch_dama:
            dama_reference = self._get_dama_reference(
                business_rule=business_rule,
                data_quality_rule=data_quality_rule,
                data_quality_dimension=data_quality_dimension,
                session=session
            )
        
        new_rule = {
            "validation_number": validation_number,
            "business_rule": business_rule,
            "data_quality_rule": data_quality_rule,
            "data_quality_dimension": data_quality_dimension,
            "Code Snowflake": code_snowflake or "",
            "Code SQL": code_sql or "",
            "Code Python": code_python or "",
            "Dama Reference": dama_reference or "",
            "raci": raci,
            "status": status,
            "created_date": now,
            "created_by": created_by,
            "updated_date": now,
            "updated_by": created_by,
            "version": "1.0",
            "notes": notes or ""
        }
        
        self.rules.append(new_rule)
        self._save_rules()
        logger.info(f"Added new rule: {validation_number}")
        return validation_number
    
    def update_rule(self, validation_number: str, updated_by: str = "System",
                   business_rule: Optional[str] = None,
                   data_quality_rule: Optional[str] = None,
                   data_quality_dimension: Optional[str] = None,
                   raci: Optional[str] = None,
                   status: Optional[str] = None,
                   notes: Optional[str] = None,
                   code_snowflake: Optional[str] = None,
                   code_sql: Optional[str] = None,
                   code_python: Optional[str] = None,
                   dama_reference: Optional[str] = None,
                   session: Optional[Any] = None,
                   auto_fetch_dama: bool = False) -> bool:
        """
        Update an existing validation rule
        
        Returns:
            True if rule was found and updated, False otherwise
        """
        for rule in self.rules:
            if rule.get('validation_number') == validation_number:
                # Update fields if provided
                if business_rule is not None:
                    rule['business_rule'] = business_rule
                if data_quality_rule is not None:
                    rule['data_quality_rule'] = data_quality_rule
                if data_quality_dimension is not None:
                    rule['data_quality_dimension'] = data_quality_dimension
                if raci is not None:
                    rule['raci'] = raci
                if status is not None:
                    rule['status'] = status
                if notes is not None:
                    rule['notes'] = notes
                if code_snowflake is not None:
                    rule['Code Snowflake'] = code_snowflake
                if code_sql is not None:
                    rule['Code SQL'] = code_sql
                if code_python is not None:
                    rule['Code Python'] = code_python
                if dama_reference is not None:
                    rule['Dama Reference'] = dama_reference
                elif auto_fetch_dama:
                    # Auto-fetch DAMA reference if rule content changed
                    current_business_rule = business_rule or rule.get('business_rule', '')
                    current_dq_rule = data_quality_rule or rule.get('data_quality_rule', '')
                    current_dimension = data_quality_dimension or rule.get('data_quality_dimension', '')
                    if current_business_rule and current_dq_rule and current_dimension:
                        rule['Dama Reference'] = self._get_dama_reference(
                            business_rule=current_business_rule,
                            data_quality_rule=current_dq_rule,
                            data_quality_dimension=current_dimension,
                            session=session
                        )
                
                # Update audit fields
                rule['updated_date'] = datetime.now().isoformat()
                rule['updated_by'] = updated_by
                
                # Increment version
                try:
                    current_version = float(rule.get('version', '1.0'))
                    rule['version'] = f"{current_version + 0.1:.1f}"
                except (ValueError, TypeError):
                    rule['version'] = "1.1"
                
                self._save_rules()
                logger.info(f"Updated rule: {validation_number}")
                return True
        
        logger.warning(f"Rule not found: {validation_number}")
        return False
    
    def get_rule(self, validation_number: str) -> Optional[Dict[str, Any]]:
        """Get a specific rule by validation number"""
        for rule in self.rules:
            if rule.get('validation_number') == validation_number:
                return rule
        return None
    
    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Get all rules"""
        return self.rules.copy()
    
    def _get_dama_reference(self, business_rule: str, data_quality_rule: str, 
                           data_quality_dimension: str, session=None) -> str:
        """
        Get DAMA DMBOK reference using RAG system
        
        Args:
            business_rule: Business rule description
            data_quality_rule: Data quality rule description
            data_quality_dimension: Data quality dimension
            session: Snowflake session (if RAG system not initialized)
        
        Returns:
            Formatted DAMA reference with chapter citations (multiple references separated by ' | ')
        """
        if not self.rag_system:
            # Try to initialize RAG system if session provided
            if session:
                RAGSystemClass = _lazy_import_rag_system()
                if RAGSystemClass is None:
                    logger.warning("RAG system not available (rag_system module not found)")
                    return ""
                try:
                    rag_database = os.getenv("SNOWFLAKE_RAG_DATABASE", "RAG_KNOWLEDGE_BASE")
                    rag_schema = os.getenv("SNOWFLAKE_RAG_SCHEMA", "PUBLIC")
                    self.rag_system = RAGSystemClass(session=session, vector_store_database=rag_database, vector_store_schema=rag_schema)
                    logger.info("RAG system initialized for DAMA references")
                except Exception as e:
                    logger.warning(f"Could not initialize RAG system for DAMA references: {e}")
                    return ""
            else:
                return ""
        
        try:
            # Build query for RAG retrieval
            query = f"{data_quality_dimension} {business_rule} {data_quality_rule} DAMA DMBOK"
            
            # Retrieve relevant chunks from DAMA DMBOK
            chunks = self.rag_system.retrieve_relevant_chunks(query=query, top_k=5)
            
            if not chunks:
                logger.debug(f"No DAMA references found for: {data_quality_dimension}")
                return ""
            
            # Format references with chapter information
            references = []
            seen_chapters = set()
            
            for chunk in chunks:
                content = chunk.get('content', '')
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', 'DAMA DMBOK 2nd Edition')
                
                # Try to extract chapter information from metadata or content
                chapter = metadata.get('chapter', '')
                page = metadata.get('page', '')
                
                # Look for chapter patterns in content
                chapter_match = re.search(r'[Cc]hapter\s+(\d+[\.\d]*)', content)
                if chapter_match and not chapter:
                    chapter = chapter_match.group(1)
                
                # Create reference string
                if chapter:
                    ref_key = f"{source} Chapter {chapter}"
                else:
                    ref_key = source
                
                # Avoid duplicates
                if ref_key in seen_chapters:
                    continue
                seen_chapters.add(ref_key)
                
                # Extract relevant excerpt (first 200 chars)
                excerpt = content[:200].strip()
                if len(content) > 200:
                    excerpt += "..."
                
                # Format reference
                if chapter and page:
                    ref = f"{ref_key}, Page {page}: {excerpt}"
                elif chapter:
                    ref = f"{ref_key}: {excerpt}"
                else:
                    ref = f"{source}: {excerpt}"
                
                references.append(ref)
            
            # Combine all references in one cell (separated by ' | ' for CSV compatibility)
            if references:
                return " | ".join(references)  # Use pipe separator for CSV compatibility
            else:
                return ""
        
        except Exception as e:
            logger.warning(f"Error retrieving DAMA reference: {e}")
            return ""
    
    def _normalize_table_name(self, table_name: str, schema_name: str) -> str:
        """
        Normalize table name to full format: DATABASE.SCHEMA.TABLE
        
        Args:
            table_name: Table name
            schema_name: Schema name (can be "DATABASE.SCHEMA" or just "SCHEMA")
        
        Returns:
            Full table name in format DATABASE.SCHEMA.TABLE
        """
        # If schema_name already includes database (has a dot), use as-is
        if '.' in schema_name:
            return f"{schema_name}.{table_name}"
        
        # Otherwise, try to get database from secrets.toml
        try:
            secrets_path = Path(".streamlit/secrets.toml")
            if secrets_path.exists():
                with open(secrets_path, 'r') as f:
                    content = f.read()
                db_match = re.search(r'database\s*=\s*["\']([^"\']+)["\']', content)
                if db_match:
                    database = db_match.group(1)
                    return f"{database}.{schema_name}.{table_name}"
        except Exception as e:
            logger.debug(f"Could not read database from secrets.toml: {e}")
        
        # Fallback: just use schema.table
        return f"{schema_name}.{table_name}"
    
    def _get_sample_data(self, session, table_name: str, schema_name: str, sample_size: int = 10) -> List[Dict]:
        """
        Get sample data from Snowflake table
        
        Args:
            session: Snowflake session
            table_name: Table name
            schema_name: Schema name (can include database, e.g., "SNOWFLAKE_SAMPLE_DATA.TPCH_SF1")
            sample_size: Number of sample rows to retrieve
        
        Returns:
            List of dictionaries representing sample rows
        """
        try:
            full_table_name = self._normalize_table_name(table_name, schema_name)
            query = f"SELECT * FROM {full_table_name} SAMPLE ({sample_size} ROWS)"
            
            logger.info(f"Sampling {sample_size} rows from {full_table_name}")
            
            if hasattr(session, 'sql'):  # Snowpark
                df = session.sql(query).to_pandas()
                return df.to_dict('records')
            else:  # Connector
                cursor = session.cursor()
                try:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                finally:
                    cursor.close()
        except Exception as e:
            logger.warning(f"Could not sample data from {schema_name}.{table_name}: {e}")
            return []
    
    def _get_data_statistics(self, session, table_name: str, schema_name: str, 
                            column_info: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Get data quality statistics from Snowflake table
        
        Args:
            session: Snowflake session
            table_name: Table name
            schema_name: Schema name (can include database)
            column_info: List of column information
        
        Returns:
            Dictionary with statistics (null counts, value ranges, etc.)
        """
        stats = {}
        try:
            full_table_name = self._normalize_table_name(table_name, schema_name)
            
            if not column_info:
                # Get column info if not provided
                desc_query = f"DESCRIBE TABLE {full_table_name}"
                if hasattr(session, 'sql'):
                    desc_df = session.sql(desc_query).to_pandas()
                    column_info = []
                    for _, row in desc_df.iterrows():
                        col_name = row.get('name', row.get('NAME', ''))
                        if col_name:
                            column_info.append({"name": str(col_name), "type": str(row.get('type', 'VARCHAR'))})
                else:
                    cursor = session.cursor()
                    try:
                        cursor.execute(desc_query)
                        columns = cursor.fetchall()
                        column_info = [{"name": col[0], "type": col[1]} for col in columns]
                    finally:
                        cursor.close()
            
            # Get statistics for each column
            for col in column_info[:20]:  # Limit to first 20 columns to avoid huge queries
                col_name = col.get('name', '')
                if not col_name:
                    continue
                
                try:
                    # Get null count and total count
                    null_query = f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        SUM(CASE WHEN "{col_name}" IS NULL THEN 1 ELSE 0 END) as null_count,
                        COUNT(DISTINCT "{col_name}") as distinct_count
                    FROM {full_table_name}
                    """
                    
                    if hasattr(session, 'sql'):
                        result_df = session.sql(null_query).to_pandas()
                        if not result_df.empty:
                            row = result_df.iloc[0]
                            stats[col_name] = {
                                "total_rows": int(row.get('TOTAL_ROWS', 0)),
                                "null_count": int(row.get('NULL_COUNT', 0)),
                                "distinct_count": int(row.get('DISTINCT_COUNT', 0)),
                                "null_percentage": (int(row.get('NULL_COUNT', 0)) / int(row.get('TOTAL_ROWS', 1))) * 100
                            }
                    else:
                        cursor = session.cursor()
                        try:
                            cursor.execute(null_query)
                            result = cursor.fetchone()
                            if result:
                                total, null_count, distinct_count = result
                                stats[col_name] = {
                                    "total_rows": total or 0,
                                    "null_count": null_count or 0,
                                    "distinct_count": distinct_count or 0,
                                    "null_percentage": ((null_count or 0) / (total or 1)) * 100
                                }
                        finally:
                            cursor.close()
                    
                    # For numeric columns, get min/max
                    col_type = str(col.get('type', '')).upper()
                    if any(t in col_type for t in ['NUMBER', 'INT', 'FLOAT', 'DECIMAL', 'NUMERIC']):
                        try:
                            range_query = f"""
                            SELECT 
                                MIN("{col_name}") as min_value,
                                MAX("{col_name}") as max_value,
                                AVG("{col_name}") as avg_value
                            FROM {full_table_name}
                            WHERE "{col_name}" IS NOT NULL
                            """
                            
                            if hasattr(session, 'sql'):
                                range_df = session.sql(range_query).to_pandas()
                                if not range_df.empty:
                                    row = range_df.iloc[0]
                                    if col_name in stats:
                                        stats[col_name].update({
                                            "min_value": float(row.get('MIN_VALUE', 0)) if row.get('MIN_VALUE') is not None else None,
                                            "max_value": float(row.get('MAX_VALUE', 0)) if row.get('MAX_VALUE') is not None else None,
                                            "avg_value": float(row.get('AVG_VALUE', 0)) if row.get('AVG_VALUE') is not None else None
                                        })
                            else:
                                cursor = session.cursor()
                                try:
                                    cursor.execute(range_query)
                                    result = cursor.fetchone()
                                    if result and col_name in stats:
                                        min_val, max_val, avg_val = result
                                        stats[col_name].update({
                                            "min_value": float(min_val) if min_val is not None else None,
                                            "max_value": float(max_val) if max_val is not None else None,
                                            "avg_value": float(avg_val) if avg_val is not None else None
                                        })
                                finally:
                                    cursor.close()
                        except Exception as e:
                            logger.debug(f"Could not get range stats for {col_name}: {e}")
                
                except Exception as e:
                    logger.debug(f"Could not get statistics for {col_name}: {e}")
                    continue
            
            logger.info(f"Retrieved statistics for {len(stats)} columns from {full_table_name}")
            return stats
        
        except Exception as e:
            logger.warning(f"Could not get data statistics from {schema_name}.{table_name}: {e}")
            return {}
    
    def analyze_and_update_with_llm(self, session, model: str, 
                                   data_context: Dict[str, Any],
                                   table_name: Optional[str] = None,
                                   schema_name: Optional[str] = None,
                                   column_info: Optional[List[Dict]] = None,
                                   include_sample_data: bool = True,
                                   include_statistics: bool = True,
                                   sample_size: int = 10) -> Dict[str, Any]:
        """
        Use LLM to analyze data and update/create validation rules
        
        Args:
            session: Snowflake session
            model: Snowflake Cortex model to use
            data_context: Context about the data (e.g., table schema, sample data, business context)
            table_name: Optional table name being analyzed
            schema_name: Optional schema name
            column_info: Optional list of column information (name, type, nullability, etc.)
            include_sample_data: Whether to query and include sample data rows (default: True)
            include_statistics: Whether to query and include data statistics (default: True)
            sample_size: Number of sample rows to retrieve (default: 10)
        
        Returns:
            Dictionary with analysis results and rules created/updated
        """
        # Use execute_cortex_query from test_models (lazy import to avoid circular dependency)
        query_func = _lazy_import_execute_cortex_query()
        if query_func is None:
            def execute_cortex_query_fallback(session, model, prompt):
                escaped_prompt = prompt.replace("'", "''")
                sql_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{escaped_prompt}') AS response"
                if hasattr(session, 'sql'):
                    result = session.sql(sql_query).collect()
                    response_raw = result[0][0] if isinstance(result[0], (list, tuple)) else result[0]
                else:
                    cursor = session.cursor()
                    try:
                        cursor.execute(sql_query)
                        result = cursor.fetchone()
                        response_raw = result[0] if result else None
                    finally:
                        cursor.close()
                if isinstance(response_raw, str):
                    try:
                        response = json.loads(response_raw)
                        if isinstance(response, dict) and "choices" in response:
                            if len(response["choices"]) > 0:
                                choice = response["choices"][0]
                                if "message" in choice:
                                    return choice["message"].get("content", "")
                                elif "messages" in choice:
                                    return choice["messages"]
                                elif "text" in choice:
                                    return choice["text"]
                        return str(response_raw)
                    except Exception:
                        return str(response_raw)
                return str(response_raw) if response_raw else ""
            
            # Use fallback
            query_func = execute_cortex_query_fallback
        
        # Prepare context for LLM
        existing_rules_summary = self._prepare_existing_rules_summary()
        
        # Get actual data from Snowflake if table/schema provided
        sample_data = []
        data_statistics = {}
        
        # If schema_name not provided but table_name is, try to get from secrets/config
        if table_name and not schema_name:
            try:
                secrets_path = Path(".streamlit/secrets.toml")
                if secrets_path.exists():
                    with open(secrets_path, 'r') as f:
                        content = f.read()
                    # Extract database and schema from secrets
                    db_match = re.search(r'database\s*=\s*["\']([^"\']+)["\']', content)
                    schema_match = re.search(r'schema\s*=\s*["\']([^"\']+)["\']', content)
                    if db_match and schema_match:
                        database = db_match.group(1)
                        schema = schema_match.group(1)
                        schema_name = f"{database}.{schema}"
                        logger.info(f"Using schema from secrets.toml: {schema_name}")
            except Exception as e:
                logger.debug(f"Could not read schema from secrets.toml: {e}")
        
        if table_name and schema_name:
            if include_sample_data:
                try:
                    sample_data = self._get_sample_data(session, table_name, schema_name, sample_size)
                    logger.info(f"Retrieved {len(sample_data)} sample rows from {schema_name}.{table_name}")
                except Exception as e:
                    logger.warning(f"Could not retrieve sample data: {e}")
            
            if include_statistics:
                try:
                    data_statistics = self._get_data_statistics(session, table_name, schema_name, column_info)
                    logger.info(f"Retrieved statistics for {len(data_statistics)} columns")
                except Exception as e:
                    logger.warning(f"Could not retrieve data statistics: {e}")
        
        # Format sample data for prompt
        sample_data_str = ""
        if sample_data:
            # Limit to first 5 rows to avoid token limits
            limited_samples = sample_data[:5]
            sample_data_str = "\nSample Data Rows:\n"
            for idx, row in enumerate(limited_samples, 1):
                # Convert row to string, limit each value length
                row_str = ", ".join([f"{k}: {str(v)[:100]}" for k, v in list(row.items())[:10]])
                sample_data_str += f"  Row {idx}: {row_str}\n"
            if len(sample_data) > 5:
                sample_data_str += f"  ... and {len(sample_data) - 5} more rows\n"
        
        # Format statistics for prompt
        statistics_str = ""
        if data_statistics:
            statistics_str = "\nData Statistics:\n"
            for col_name, stats in list(data_statistics.items())[:15]:  # Limit to 15 columns
                stats_lines = [f"  {col_name}:"]
                if 'null_count' in stats:
                    stats_lines.append(f"    - Null values: {stats['null_count']}/{stats['total_rows']} ({stats.get('null_percentage', 0):.1f}%)")
                if 'distinct_count' in stats:
                    stats_lines.append(f"    - Distinct values: {stats['distinct_count']}")
                if 'min_value' in stats and stats['min_value'] is not None:
                    stats_lines.append(f"    - Range: {stats['min_value']} to {stats['max_value']}")
                    if 'avg_value' in stats and stats['avg_value'] is not None:
                        stats_lines.append(f"    - Average: {stats['avg_value']:.2f}")
                statistics_str += "\n".join(stats_lines) + "\n"
        
        # Build comprehensive prompt
        analysis_prompt = f"""You are a data quality expert analyzing actual data from a Snowflake database to create and update validation rules.

EXISTING VALIDATION RULES:
{existing_rules_summary}

DATA CONTEXT:
Table: {table_name or 'Not specified'}
Schema: {schema_name or 'Not specified'}

Column Information:
{self._format_column_info(column_info) if column_info else 'Not provided'}
{sample_data_str}
{statistics_str}
Additional Context:
{json.dumps(data_context, indent=2) if data_context else 'None'}

TASK:
Analyze the ACTUAL DATA (sample rows and statistics) along with the schema to:
1. Identify data quality issues by examining the sample data and statistics
2. Create validation rules based on patterns you observe in the actual data
3. Identify missing or incomplete data based on null counts and percentages
4. Detect data anomalies, outliers, or inconsistencies in the sample data
5. Map each rule to appropriate data quality dimensions
6. Assign RACI (Responsible, Accountable, Consulted, Informed) based on business impact
7. Update existing rules if the actual data reveals new information or contradicts existing assumptions

CRITICAL REQUIREMENT - MINIMUM RULES PER DIMENSION:
You MUST generate AT LEAST 5 validation rules for EACH of the following data quality dimensions:
- Completeness: Rules checking for missing/null values, required fields, data coverage
- Accuracy: Rules checking for correct values, valid formats, data correctness

For other dimensions (Consistency, Timeliness, Validity, Uniqueness, Integrity, Conformity, Precision, Currency), generate at least 2-3 rules each if applicable to the data.

IMPORTANT: Base your analysis on the ACTUAL DATA provided (sample rows and statistics), not just the schema. Look for:
- Null value patterns (high null percentages may indicate data quality issues) → Completeness rules
- Value ranges that seem unusual or need validation → Accuracy rules
- Data inconsistencies in sample rows → Consistency rules
- Missing required fields based on business logic → Completeness rules
- Format issues visible in sample data → Accuracy/Validity rules
- Relationships or dependencies between columns visible in the data → Integrity rules
- Duplicate values or uniqueness issues → Uniqueness rules
- Date/timestamp issues → Timeliness rules

For COMPLETENESS dimension, create rules such as:
- Required field checks (columns that should never be null)
- Percentage completeness thresholds
- Mandatory field combinations
- Data coverage rules
- Missing data pattern detection

For ACCURACY dimension, create rules such as:
- Format validation (email, phone, date formats)
- Range validation (numeric values within expected bounds)
- Value correctness (valid codes, valid references)
- Calculation accuracy (derived fields match expected formulas)
- Data type accuracy (values match declared types)

DATA QUALITY DIMENSIONS (choose the most appropriate):
{chr(10).join(f"- {dim}" for dim in DATA_QUALITY_DIMENSIONS)}

RACI VALUES:
- Responsible: Person/team who performs the work
- Accountable: Person/team ultimately answerable for the outcome
- Consulted: Person/team who provides input
- Informed: Person/team who needs to be notified

OUTPUT FORMAT (JSON):
You MUST provide ALL required fields for each rule. The CSV has these columns that need to be populated:

REQUIRED FIELDS (must be provided for every new rule):
{{
    "new_rules": [
        {{
            "business_rule": "REQUIRED - Clear business requirement description explaining WHY this rule exists and what business problem it solves. Example: 'Customer email addresses must be valid to ensure successful communication and order confirmations'",
            "data_quality_rule": "REQUIRED - Specific technical validation rule describing HOW to validate the data. Example: 'Email format validation using RFC 5322 standard and uniqueness check across all customer records'",
            "data_quality_dimension": "REQUIRED - Must be one of: {', '.join(DATA_QUALITY_DIMENSIONS)}. Choose the dimension that best fits the rule (e.g., 'Validity' for format checks, 'Completeness' for null checks, 'Uniqueness' for duplicate checks)",
            "code_snowflake": "OPTIONAL but recommended - Snowflake SQL code to implement this validation rule. Use Snowflake-specific functions and syntax. Example: 'SELECT COUNT(*) FROM table WHERE column IS NULL'",
            "code_sql": "OPTIONAL but recommended - Standard SQL code to implement this validation rule. Use ANSI SQL syntax that works across databases. Example: 'SELECT COUNT(*) FROM table WHERE column IS NULL'",
            "code_python": "OPTIONAL but recommended - Python code to implement this validation rule. Can use libraries like pandas, great_expectations, or custom validation logic. Example: 'df[df[\"column\"].isnull()].shape[0]'",
            "dama_reference": "OPTIONAL but recommended - DAMA DMBOK 2nd Edition reference with chapter citations. Format: 'DAMA DMBOK 2nd Edition Chapter X: [excerpt] | Chapter Y: [excerpt]'. Multiple related references should be in the same cell separated by ' | '. Include chapter numbers and brief excerpts. If you don't provide this, the system will automatically fetch DAMA references using RAG.",
            "raci": "REQUIRED - Must be exactly one of: Responsible, Accountable, Consulted, or Informed. Choose based on who should own/perform this validation",
            "notes": "OPTIONAL but recommended - Additional context, rationale, or specific examples from the data that support this rule"
        }}
    ],
    "updates": [
        {{
            "validation_number": "REQUIRED for updates - The existing validation number (e.g., 'VR-001')",
            "business_rule": "OPTIONAL - Updated business rule if changed",
            "data_quality_rule": "OPTIONAL - Updated data quality rule if changed",
            "data_quality_dimension": "OPTIONAL - Updated dimension if changed",
            "code_snowflake": "OPTIONAL - Updated Snowflake code if changed",
            "code_sql": "OPTIONAL - Updated SQL code if changed",
            "code_python": "OPTIONAL - Updated Python code if changed",
            "dama_reference": "OPTIONAL - Updated DAMA DMBOK reference if changed",
            "raci": "OPTIONAL - Updated RACI if changed",
            "notes": "OPTIONAL - Reason for update or additional context"
        }}
    ],
    "analysis_summary": "Brief summary of findings and recommendations"
}}

FIELD REQUIREMENTS:
- business_rule: MUST be a clear, business-focused description (2-3 sentences). Explain the business need, not just the technical check.
- data_quality_rule: MUST be a specific, technical description of the validation logic. Include what is being checked and how.
- data_quality_dimension: MUST be exactly one of the listed dimensions. Choose the most appropriate one.
- code_snowflake: RECOMMENDED - Generate executable Snowflake SQL code that implements this validation. Use actual table/column names from the data context. Make it production-ready.
- code_sql: RECOMMENDED - Generate standard ANSI SQL code that implements this validation. Use actual table/column names from the data context. Make it portable across databases.
- code_python: RECOMMENDED - Generate Python code that implements this validation. Can use pandas, great_expectations, or custom logic. Use actual column names from the data context.
- raci: MUST be exactly one of: Responsible, Accountable, Consulted, or Informed (case-sensitive).
- notes: Should include specific examples from the actual data that support this rule, or explain why this rule is important.

CODE GENERATION GUIDELINES:
- Use the actual table name and column names from the DATA CONTEXT provided above
- Generate executable, production-ready code
- Include comments in code where helpful
- For Snowflake: Use Snowflake-specific functions (e.g., TRY_TO_NUMBER, REGEXP_LIKE, etc.)
- For SQL: Use standard SQL that works across databases
- For Python: Use appropriate libraries (pandas, great_expectations) based on the validation type

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON starting with {{ and ending with }}. Do NOT wrap in markdown code blocks. Do NOT add explanations before or after the JSON.
- The response must be parseable as JSON - start directly with {{"new_rules": [ ... ]}}
- ALL required fields (business_rule, data_quality_rule, data_quality_dimension, raci) MUST be provided for every new rule
- Use proper English grammar and professional language
- Be specific and actionable in rule descriptions
- Base rules on ACTUAL DATA patterns you observe in the sample rows and statistics
- Map rules to the most appropriate data quality dimension
- Consider business impact when assigning RACI
- If updating existing rules, provide validation_number
- If creating new rules, do NOT include validation_number (it will be auto-generated)
- Ensure all field values are properly formatted strings (no null values)
- If a field is truly not applicable, use an empty string "" rather than omitting it
- IMPORTANT: Your response must be valid JSON that can be parsed by json.loads(). Start with {{ and end with }}

MINIMUM RULE GENERATION REQUIREMENTS:
- Generate AT LEAST 5 rules for "Completeness" dimension
- Generate AT LEAST 5 rules for "Accuracy" dimension
- Generate 2-3 rules for other applicable dimensions (Consistency, Timeliness, Validity, Uniqueness, Integrity, Conformity, Precision, Currency)
- Total minimum: 15-20 validation rules should be generated
- Each rule must be unique and address different aspects of data quality
- Use the actual column names, table names, and data patterns from the DATA CONTEXT provided above
"""
        
        try:
            logger.info("Calling LLM to analyze data and generate/update rules...")
            llm_response_raw = query_func(session, model, analysis_prompt)
            
            # Log raw response for debugging (first 500 chars)
            if isinstance(llm_response_raw, str):
                logger.debug(f"LLM raw response (first 500 chars): {llm_response_raw[:500]}")
            else:
                logger.debug(f"LLM raw response type: {type(llm_response_raw)}")
            
            # Parse LLM response
            if isinstance(llm_response_raw, str):
                # Try to extract JSON from response
                llm_response = self._extract_json_from_response(llm_response_raw)
                if not llm_response:
                    # Log the full response if extraction failed
                    logger.error(f"Failed to extract JSON. Full response (first 2000 chars): {llm_response_raw[:2000]}")
            else:
                llm_response = llm_response_raw
            
            if not isinstance(llm_response, dict):
                logger.error(f"LLM response is not a valid dictionary. Type: {type(llm_response)}, Value: {str(llm_response)[:500]}")
                # Try one more time with a different approach - maybe it's a list wrapped in something
                if isinstance(llm_response_raw, str):
                    # Try to find and parse just the new_rules array
                    rules_match = re.search(r'"new_rules"\s*:\s*\[(.*?)\]', llm_response_raw, re.DOTALL)
                    if rules_match:
                        try:
                            # Try to reconstruct a valid JSON
                            rules_str = rules_match.group(0)
                            reconstructed = "{" + rules_str + ", \"updates\": [], \"analysis_summary\": \"\"}"
                            llm_response = json.loads(reconstructed)
                            logger.info("Successfully reconstructed JSON from new_rules array")
                        except Exception:
                            pass
                
                if not isinstance(llm_response, dict):
                    return {
                        "success": False,
                        "error": "Invalid LLM response format - could not parse as JSON dictionary",
                        "rules_created": 0,
                        "rules_updated": 0,
                        "raw_response_preview": str(llm_response_raw)[:1000] if isinstance(llm_response_raw, str) else str(type(llm_response_raw))
                    }
            
            # Process new rules
            rules_created = []
            dimension_counts = {}
            
            new_rules_list = llm_response.get('new_rules', [])
            logger.info(f"Processing {len(new_rules_list)} rules from LLM response")
            if new_rules_list and len(new_rules_list) > 0:
                logger.debug(f"First rule structure: {json.dumps(new_rules_list[0], indent=2)[:500]}")
            
            for idx, new_rule in enumerate(new_rules_list):
                try:
                    # Validate required fields
                    business_rule = new_rule.get('business_rule', '').strip()
                    data_quality_rule = new_rule.get('data_quality_rule', '').strip()
                    data_quality_dimension = new_rule.get('data_quality_dimension', '').strip()
                    raci = new_rule.get('raci', '').strip()
                    
                    # Log rule details for debugging
                    logger.debug(f"Rule {idx+1} fields - business_rule: {bool(business_rule)}, data_quality_rule: {bool(data_quality_rule)}, dimension: {data_quality_dimension}, raci: {raci}")
                    
                    # Validate required fields are present
                    if not business_rule:
                        logger.warning(f"Skipping rule {idx+1}: missing business_rule. Rule keys: {list(new_rule.keys())}")
                        continue
                    if not data_quality_rule:
                        logger.warning("Skipping rule: missing data_quality_rule")
                        continue
                    if not data_quality_dimension:
                        logger.warning("Skipping rule: missing data_quality_dimension, using default 'Completeness'")
                        data_quality_dimension = 'Completeness'
                    if not raci:
                        logger.warning("Skipping rule: missing raci, using default 'Responsible'")
                        raci = 'Responsible'
                    
                    # Validate data_quality_dimension is valid
                    if data_quality_dimension not in DATA_QUALITY_DIMENSIONS:
                        logger.warning(f"Invalid dimension '{data_quality_dimension}', using default 'Completeness'")
                        data_quality_dimension = 'Completeness'
                    
                    # Validate RACI is valid
                    if raci not in RACI_VALUES:
                        logger.warning(f"Invalid RACI '{raci}', using default 'Responsible'")
                        raci = 'Responsible'
                    
                    validation_number = self.add_rule(
                        business_rule=business_rule,
                        data_quality_rule=data_quality_rule,
                        data_quality_dimension=data_quality_dimension,
                        raci=raci,
                        created_by="LLM Analysis",
                        status="Draft",
                        notes=new_rule.get('notes', '').strip(),
                        code_snowflake=new_rule.get('code_snowflake', '').strip(),
                        code_sql=new_rule.get('code_sql', '').strip(),
                        code_python=new_rule.get('code_python', '').strip(),
                        dama_reference=new_rule.get('dama_reference', '').strip(),
                        session=session,
                        auto_fetch_dama=True  # Auto-fetch if LLM didn't provide
                    )
                    rules_created.append(validation_number)
                    
                    # Track dimension counts
                    dimension_counts[data_quality_dimension] = dimension_counts.get(data_quality_dimension, 0) + 1
                    
                    logger.info(f"Created rule {validation_number} [{data_quality_dimension}]: {business_rule[:50]}...")
                except Exception as e:
                    logger.error(f"Error creating rule: {e}")
            
            # Validate minimum rule requirements
            completeness_count = dimension_counts.get('Completeness', 0)
            accuracy_count = dimension_counts.get('Accuracy', 0)
            
            if completeness_count < 5:
                logger.warning(f"Only {completeness_count} Completeness rules generated (minimum 5 required)")
            if accuracy_count < 5:
                logger.warning(f"Only {accuracy_count} Accuracy rules generated (minimum 5 required)")
            
            if completeness_count >= 5 and accuracy_count >= 5:
                logger.info(f"✓ Minimum rule requirements met: {completeness_count} Completeness, {accuracy_count} Accuracy")
            
            # Log dimension summary
            if dimension_counts:
                dimension_summary = ", ".join([f"{dim}: {count}" for dim, count in sorted(dimension_counts.items())])
                logger.info(f"Rules by dimension: {dimension_summary}")
            
            # Process updates
            rules_updated = []
            for update in llm_response.get('updates', []):
                validation_number = update.get('validation_number')
                if not validation_number:
                    continue
                
                try:
                    success = self.update_rule(
                        validation_number=validation_number,
                        updated_by="LLM Analysis",
                        business_rule=update.get('business_rule'),
                        data_quality_rule=update.get('data_quality_rule'),
                        data_quality_dimension=update.get('data_quality_dimension'),
                        raci=update.get('raci'),
                        notes=update.get('notes'),
                        code_snowflake=update.get('code_snowflake'),
                        code_sql=update.get('code_sql'),
                        code_python=update.get('code_python'),
                        dama_reference=update.get('dama_reference'),
                        session=session,
                        auto_fetch_dama=True  # Auto-fetch if LLM didn't provide
                    )
                    if success:
                        rules_updated.append(validation_number)
                except Exception as e:
                    logger.error(f"Error updating rule {validation_number}: {e}")
            
            # Calculate dimension statistics
            dimension_stats = {}
            for rule_num in rules_created:
                rule = self.get_rule(rule_num)
                if rule:
                    dim = rule.get('data_quality_dimension', 'Unknown')
                    dimension_stats[dim] = dimension_stats.get(dim, 0) + 1
            
            return {
                "success": True,
                "rules_created": len(rules_created),
                "rules_updated": len(rules_updated),
                "validation_numbers_created": rules_created,
                "validation_numbers_updated": rules_updated,
                "dimension_counts": dimension_stats,
                "completeness_count": dimension_stats.get('Completeness', 0),
                "accuracy_count": dimension_stats.get('Accuracy', 0),
                "minimum_requirements_met": {
                    "completeness": dimension_stats.get('Completeness', 0) >= 5,
                    "accuracy": dimension_stats.get('Accuracy', 0) >= 5
                },
                "analysis_summary": llm_response.get('analysis_summary', ''),
                "llm_response": llm_response
            }
        
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "rules_created": 0,
                "rules_updated": 0
            }
    
    def _prepare_existing_rules_summary(self) -> str:
        """Prepare summary of existing rules for LLM context"""
        if not self.rules:
            return "No existing rules."
        
        summary_lines = []
        for rule in self.rules[:20]:  # Limit to first 20 for context
            summary_lines.append(
                f"- {rule.get('validation_number', 'N/A')}: "
                f"Business Rule: {rule.get('business_rule', 'N/A')[:100]} | "
                f"DQ Rule: {rule.get('data_quality_rule', 'N/A')[:100]} | "
                f"Dimension: {rule.get('data_quality_dimension', 'N/A')} | "
                f"RACI: {rule.get('raci', 'N/A')}"
            )
        
        if len(self.rules) > 20:
            summary_lines.append(f"... and {len(self.rules) - 20} more rules")
        
        return "\n".join(summary_lines) if summary_lines else "No existing rules."
    
    def _format_column_info(self, column_info: List[Dict]) -> str:
        """Format column information for LLM prompt"""
        if not column_info:
            return "No column information provided"
        
        lines = []
        for col in column_info:
            col_name = col.get('name', 'Unknown')
            col_type = col.get('type', 'Unknown')
            nullable = col.get('nullable', 'Unknown')
            lines.append(f"- {col_name}: {col_type} (Nullable: {nullable})")
        
        return "\n".join(lines)
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON from LLM response (may contain markdown or extra text)"""
        if not response or not isinstance(response, str):
            logger.error("Empty or invalid response")
            return {}
        
        # Clean up response - remove leading/trailing whitespace
        response = response.strip()
        
        # Try to extract JSON from markdown code blocks (```json ... ``` or ``` ... ```)
        code_block_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
        ]
        
        for pattern in code_block_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict):
                        logger.debug("Successfully extracted JSON from code block")
                        return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from code block: {e}")
                    continue
        
        # Try to find JSON object (look for { ... } with balanced braces)
        # Use a more sophisticated approach to find the largest valid JSON object
        brace_count = 0
        start_idx = -1
        max_obj = None
        max_obj_str = None
        
        for i, char in enumerate(response):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    candidate = response[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict) and (max_obj is None or len(candidate) > len(max_obj_str or '')):
                            max_obj = parsed
                            max_obj_str = candidate
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        
        if max_obj:
            logger.debug("Successfully extracted JSON object using brace matching")
            return max_obj
        
        # Try to find JSON array
        bracket_count = 0
        start_idx = -1
        
        for i, char in enumerate(response):
            if char == '[':
                if bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0 and start_idx >= 0:
                    candidate = response[start_idx:i+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, list):
                            # Wrap array in a dict with 'new_rules' key
                            logger.debug("Found JSON array, wrapping in dict")
                            return {"new_rules": parsed}
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
        
        # Last resort: try parsing entire response
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                logger.debug("Successfully parsed entire response as JSON")
                return parsed
        except json.JSONDecodeError:
            pass
        
        # If all else fails, log the response for debugging
        logger.error(f"Could not extract JSON from LLM response. Response length: {len(response)}")
        logger.debug(f"Response preview (first 1000 chars): {response[:1000]}")
        return {}
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all rules as dictionary for reporting"""
        return {
            "total_rules": len(self.rules),
            "rules": self.rules,
            "csv_path": str(self.csv_path),
            "export_date": datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the rules"""
        stats = {
            "total_rules": len(self.rules),
            "by_dimension": {},
            "by_raci": {},
            "by_status": {}
        }
        
        for rule in self.rules:
            dimension = rule.get('data_quality_dimension', 'Unknown')
            raci = rule.get('raci', 'Unknown')
            status = rule.get('status', 'Unknown')
            
            stats['by_dimension'][dimension] = stats['by_dimension'].get(dimension, 0) + 1
            stats['by_raci'][raci] = stats['by_raci'].get(raci, 0) + 1
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        return stats
