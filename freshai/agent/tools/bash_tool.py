"""Bash execution tool for agents."""
import subprocess
import logging
import shlex
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BashTool:
    """Tool for executing bash commands."""

    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 10000,
        working_dir: Optional[str] = None,
        safe_mode: bool = True
    ):
        """Initialize the bash tool.

        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum output length to return
            working_dir: Working directory for commands
            safe_mode: If True, block potentially dangerous commands
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.working_dir = working_dir or os.getcwd()
        self.safe_mode = safe_mode

        # List of dangerous commands to block in safe mode
        self.dangerous_commands = [
            "rm -rf",
            "dd",
            "format",
            "mkfs",
            "> /dev/",
            "sudo rm",
            "chmod 777 /",
            ":(){ :|:& };:",  # Fork bomb
        ]

    def execute(self, command: str) -> str:
        """Execute a bash command.

        Args:
            command: The bash command to execute

        Returns:
            Command output or error message
        """
        logger.info(f"Executing bash command: {command}")

        # Safety check
        if self.safe_mode:
            lower_cmd = command.lower()
            for dangerous in self.dangerous_commands:
                if dangerous.lower() in lower_cmd:
                    error = f"Blocked potentially dangerous command: {command}"
                    logger.warning(error)
                    return f"Error: {error}"

        try:
            # Parse command
            # Use shell=True for complex commands with pipes, redirections, etc.
            use_shell = any(char in command for char in ['|', '>', '<', '&', ';'])

            if use_shell:
                logger.debug("Using shell mode for complex command")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.working_dir
                )
            else:
                # Parse command for simple commands
                cmd_parts = shlex.split(command)
                logger.debug(f"Command parts: {cmd_parts}")

                result = subprocess.run(
                    cmd_parts,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.working_dir
                )

            # Combine stdout and stderr
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                if output:
                    output += "\n"
                output += f"STDERR: {result.stderr}"

            # Truncate if too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (output truncated)"

            # Add return code if non-zero
            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"

            logger.debug(f"Command output (first 500 chars): {output[:500]}")
            return output if output else "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            error = f"Command timed out after {self.timeout} seconds"
            logger.error(error)
            return f"Error: {error}"
        except Exception as e:
            error = f"Failed to execute command: {str(e)}"
            logger.error(error)
            return f"Error: {error}"

    def __call__(self, command: str) -> str:
        """Allow the tool to be called as a function."""
        return self.execute(command)


def create_bash_tool(
    timeout: int = 30,
    max_output_length: int = 10000,
    working_dir: Optional[str] = None,
    safe_mode: bool = True
) -> Dict[str, Any]:
    """Create a bash tool specification for agents.

    Args:
        timeout: Maximum execution time in seconds
        max_output_length: Maximum output length to return
        working_dir: Working directory for commands
        safe_mode: If True, block potentially dangerous commands

    Returns:
        Tool specification with function and metadata
    """
    tool = BashTool(timeout, max_output_length, working_dir, safe_mode)

    return {
        "name": "bash",
        "description": "Execute bash commands. Use this to run system commands, check files, install packages, etc.",
        "function": tool,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                }
            },
            "required": ["command"]
        }
    }