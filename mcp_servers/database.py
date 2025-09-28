#!/usr/bin/env python3
"""
Simple MCP server that provides database operations.
This is a basic implementation using SQLite for demonstration purposes.
"""

import json
import sys
import sqlite3
import os
from typing import Dict, Any, List
from pathlib import Path


class DatabaseMCPServer:
    """Simple MCP server for database operations."""
    
    def __init__(self):
        self.tools = {
            "execute_query": {
                "name": "execute_query",
                "description": "Execute a SQL query on a database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_path": {
                            "type": "string",
                            "description": "Path to the SQLite database file"
                        },
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "parameters": {
                            "type": "array",
                            "description": "Parameters for the SQL query",
                            "default": []
                        }
                    },
                    "required": ["database_path", "query"]
                }
            },
            "list_tables": {
                "name": "list_tables",
                "description": "List all tables in a database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_path": {
                            "type": "string",
                            "description": "Path to the SQLite database file"
                        }
                    },
                    "required": ["database_path"]
                }
            },
            "describe_table": {
                "name": "describe_table",
                "description": "Get schema information for a table",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_path": {
                            "type": "string",
                            "description": "Path to the SQLite database file"
                        },
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table to describe"
                        }
                    },
                    "required": ["database_path", "table_name"]
                }
            },
            "create_sample_database": {
                "name": "create_sample_database",
                "description": "Create a sample database for testing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "database_path": {
                            "type": "string",
                            "description": "Path where to create the sample database"
                        }
                    },
                    "required": ["database_path"]
                }
            }
        }
    
    def handle_tools_list(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": list(self.tools.values())
            }
        }
    
    def handle_tools_call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            if tool_name == "execute_query":
                result = self._execute_query(arguments)
            elif tool_name == "list_tables":
                result = self._list_tables(arguments)
            elif tool_name == "describe_table":
                result = self._describe_table(arguments)
            elif tool_name == "create_sample_database":
                result = self._create_sample_database(arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Tool execution failed: {str(e)}"
                }
            }
    
    def _execute_query(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a SQL query."""
        database_path = arguments["database_path"]
        query = arguments["query"]
        parameters = arguments.get("parameters", [])
        
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database file does not exist: {database_path}")
        
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()
            
            cursor.execute(query, parameters)
            
            # Determine if this is a SELECT query or modification query
            query_type = query.strip().upper().split()[0]
            
            if query_type == "SELECT":
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                # Convert rows to dictionaries
                result_rows = []
                for row in rows:
                    result_rows.append(dict(row))
                
                return {
                    "query": query,
                    "query_type": "SELECT",
                    "columns": columns,
                    "rows": result_rows,
                    "row_count": len(result_rows)
                }
            else:
                # For INSERT, UPDATE, DELETE, etc.
                conn.commit()
                affected_rows = cursor.rowcount
                
                return {
                    "query": query,
                    "query_type": query_type,
                    "affected_rows": affected_rows,
                    "success": True
                }
    
    def _list_tables(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List all tables in the database."""
        database_path = arguments["database_path"]
        
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database file does not exist: {database_path}")
        
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        
        return {
            "database_path": database_path,
            "tables": tables,
            "table_count": len(tables)
        }
    
    def _describe_table(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get schema information for a table."""
        database_path = arguments["database_path"]
        table_name = arguments["table_name"]
        
        if not os.path.exists(database_path):
            raise FileNotFoundError(f"Database file does not exist: {database_path}")
        
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if not cursor.fetchone():
                raise ValueError(f"Table '{table_name}' does not exist")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Format column information
            column_info = []
            for col in columns:
                column_info.append({
                    "name": col[1],
                    "type": col[2],
                    "not_null": bool(col[3]),
                    "default_value": col[4],
                    "primary_key": bool(col[5])
                })
        
        return {
            "database_path": database_path,
            "table_name": table_name,
            "columns": column_info,
            "column_count": len(column_info),
            "row_count": row_count
        }
    
    def _create_sample_database(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sample database for testing."""
        database_path = arguments["database_path"]
        
        # Ensure directory exists
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(database_path) as conn:
            cursor = conn.cursor()
            
            # Create sample tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS investigators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    badge_number TEXT UNIQUE NOT NULL,
                    department TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_number TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'open',
                    priority TEXT DEFAULT 'medium',
                    assigned_investigator_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (assigned_investigator_id) REFERENCES investigators (id)
                );
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER NOT NULL,
                    item_number TEXT NOT NULL,
                    description TEXT NOT NULL,
                    type TEXT NOT NULL,
                    location_found TEXT,
                    collected_by INTEGER,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES cases (id),
                    FOREIGN KEY (collected_by) REFERENCES investigators (id)
                );
            ''')
            
            # Insert sample data
            cursor.execute('''
                INSERT OR IGNORE INTO investigators (name, badge_number, department, email) VALUES
                ('John Smith', 'INV001', 'Homicide', 'j.smith@police.gov'),
                ('Sarah Johnson', 'INV002', 'Fraud', 's.johnson@police.gov'),
                ('Mike Davis', 'INV003', 'Cybercrime', 'm.davis@police.gov');
            ''')
            
            cursor.execute('''
                INSERT OR IGNORE INTO cases (case_number, title, description, status, priority, assigned_investigator_id) VALUES
                ('CASE001', 'Suspicious Activity Report', 'Investigation of unusual financial transactions', 'open', 'high', 2),
                ('CASE002', 'Data Breach Investigation', 'Corporate network intrusion analysis', 'in_progress', 'high', 3),
                ('CASE003', 'Missing Person', 'Individual reported missing 3 days ago', 'open', 'medium', 1);
            ''')
            
            cursor.execute('''
                INSERT OR IGNORE INTO evidence (case_id, item_number, description, type, location_found, collected_by) VALUES
                (1, 'EVID001-001', 'Bank statement showing irregular transfers', 'document', 'Suspect residence', 2),
                (1, 'EVID001-002', 'USB drive with encrypted files', 'digital', 'Suspect office', 2),
                (2, 'EVID002-001', 'Server logs showing unauthorized access', 'digital', 'Company server room', 3),
                (3, 'EVID003-001', 'Last known photograph', 'photograph', 'Family home', 1);
            ''')
            
            conn.commit()
        
        return {
            "database_path": database_path,
            "success": True,
            "message": "Sample database created successfully",
            "tables_created": ["investigators", "cases", "evidence"],
            "sample_data": "Added sample investigators, cases, and evidence records"
        }
    
    def run(self):
        """Run the MCP server."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    method = request.get("method")
                    
                    if method == "tools/list":
                        response = self.handle_tools_list(request)
                    elif method == "tools/call":
                        response = self.handle_tools_call(request)
                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "id": request.get("id"),
                            "error": {
                                "code": -32601,
                                "message": f"Unknown method: {method}"
                            }
                        }
                    
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {e}"
                        }
                    }
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Server error: {e}", file=sys.stderr)


if __name__ == "__main__":
    server = DatabaseMCPServer()
    server.run()