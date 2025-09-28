"""Command-line interface for FreshAI."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import InvestigatorAgent
from .config import Config
from .utils import setup_logging, validate_case_id


app = typer.Typer(
    name="freshai",
    help="FreshAI - AI Assistant for Crime Investigators",
    add_completion=False
)
console = Console()


def create_agent(config_path: Optional[str] = None) -> InvestigatorAgent:
    """Create and return an investigator agent."""
    if config_path:
        config = Config.load_from_env(config_path)
    else:
        config = Config.load_from_env()
    
    setup_logging(config.log_level)
    return InvestigatorAgent(config)


@app.command()
def init(
    directory: str = typer.Argument(".", help="Directory to initialize FreshAI in"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files")
):
    """Initialize FreshAI in the specified directory."""
    init_dir = Path(directory).resolve()
    
    console.print(f"[blue]Initializing FreshAI in {init_dir}[/blue]")
    
    # Create directories
    directories = ["evidence", "cases", "config"]
    for dir_name in directories:
        dir_path = init_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        console.print(f"Created directory: {dir_path}")
    
    # Create sample .env file
    env_file = init_dir / ".env"
    if not env_file.exists() or force:
        env_content = """# FreshAI Configuration
FRESHAI_DEBUG=false
FRESHAI_LOG_LEVEL=INFO
FRESHAI_ENABLE_VISION=true
FRESHAI_ENABLE_TOOLS=true
FRESHAI_MAX_CONTEXT_LENGTH=8192

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_DEFAULT_LLM=llama2
OLLAMA_DEFAULT_VLM=llava

# Storage Paths
FRESHAI_EVIDENCE_STORAGE_PATH=./evidence
FRESHAI_CASE_STORAGE_PATH=./cases

# Transformers Configuration
TRANSFORMERS_CACHE_DIR=./models
TRANSFORMERS_DEVICE=auto
"""
        env_file.write_text(env_content)
        console.print(f"Created configuration file: {env_file}")
    
    console.print("[green]FreshAI initialized successfully![/green]")


@app.command()
def start_case(
    case_id: str = typer.Argument(..., help="Unique case identifier"),
    description: str = typer.Option("", "--description", "-d", help="Case description"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Start a new investigation case."""
    async def _start_case():
        agent = create_agent(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.add_task("Starting case...", total=None)
            case_info = await agent.start_case(validate_case_id(case_id), description)
        
        console.print(Panel(
            f"[green]Case started successfully![/green]\n\n"
            f"Case ID: {case_info['case_id']}\n"
            f"Description: {case_info.get('description', 'N/A')}\n"
            f"Start Time: {case_info['start_time']}\n"
            f"Status: {case_info['status']}",
            title="New Case Created"
        ))
        
        await agent.cleanup()
    
    asyncio.run(_start_case())


@app.command()
def analyze(
    case_id: str = typer.Argument(..., help="Case identifier"),
    evidence_path: str = typer.Argument(..., help="Path to evidence file"),
    evidence_type: str = typer.Option("auto", "--type", "-t", help="Evidence type (auto, image, text, document)"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Analyze evidence for a case."""
    async def _analyze():
        agent = create_agent(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.add_task("Analyzing evidence...", total=None)
            result = await agent.analyze_evidence(
                validate_case_id(case_id), 
                evidence_path, 
                evidence_type
            )
        
        console.print(Panel(
            f"[green]Evidence analyzed successfully![/green]\n\n"
            f"Evidence: {result['evidence_path']}\n"
            f"Type: {result['evidence_type']}\n"
            f"Analysis Time: {result['analysis_timestamp']}",
            title="Evidence Analysis Complete"
        ))
        
        # Show key findings
        if "results" in result and not result["results"].get("error"):
            console.print("\n[bold]Key Findings:[/bold]")
            
            results = result["results"]
            if "vlm_analysis" in results:
                vlm = results["vlm_analysis"]
                if "description" in vlm:
                    console.print(f"Visual Analysis: {vlm['description'][:200]}...")
            
            if "pattern_analysis" in results:
                pattern = results["pattern_analysis"]
                if "suspicious_keywords" in pattern and pattern["suspicious_keywords"]:
                    console.print(f"Suspicious Keywords: {list(pattern['suspicious_keywords'].keys())}")
        
        await agent.cleanup()
    
    asyncio.run(_analyze())


@app.command()
def ask(
    case_id: str = typer.Argument(..., help="Case identifier"),
    question: str = typer.Argument(..., help="Question to ask about the case"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Ask a question about a case."""
    async def _ask():
        agent = create_agent(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.add_task("Processing question...", total=None)
            result = await agent.ask_question(validate_case_id(case_id), question)
        
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(Panel(
                result["answer"],
                title=f"Answer for Case {case_id}",
                subtitle=f"Model: {result.get('model', 'Unknown')}"
            ))
        
        await agent.cleanup()
    
    asyncio.run(_ask())


@app.command()
def report(
    case_id: str = typer.Argument(..., help="Case identifier"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Generate a case report."""
    async def _report():
        agent = create_agent(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.add_task("Generating report...", total=None)
            report_data = await agent.generate_case_report(validate_case_id(case_id))
        
        if "error" in report_data:
            console.print(f"[red]Error generating report: {report_data['error']}[/red]")
        else:
            console.print(Panel(
                report_data["report_content"],
                title=f"Investigation Report - Case {case_id}",
                subtitle=f"Generated: {report_data['report_generated']}"
            ))
            
            if output_file:
                output_path = Path(output_file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                console.print(f"\n[green]Report saved to: {output_path}[/green]")
        
        await agent.cleanup()
    
    asyncio.run(_report())


@app.command()
def status(
    case_id: str = typer.Argument(..., help="Case identifier"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Get case status."""
    async def _status():
        agent = create_agent(config_path)
        await agent.initialize()
        
        try:
            case_info = agent.get_case_status(validate_case_id(case_id))
            
            table = Table(title=f"Case Status: {case_id}")
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            
            table.add_row("Case ID", case_info['case_id'])
            table.add_row("Description", case_info.get('description', 'N/A'))
            table.add_row("Status", case_info['status'])
            table.add_row("Start Time", case_info['start_time'])
            table.add_row("Evidence Count", str(case_info['evidence_count']))
            table.add_row("Questions Asked", str(len(case_info.get('questions', []))))
            
            console.print(table)
            
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
        
        await agent.cleanup()
    
    asyncio.run(_status())


@app.command()
def list_cases(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """List active cases."""
    async def _list():
        agent = create_agent(config_path)
        await agent.initialize()
        
        active_cases = agent.list_active_cases()
        
        if not active_cases:
            console.print("[yellow]No active cases found.[/yellow]")
        else:
            table = Table(title="Active Cases")
            table.add_column("Case ID", style="cyan", no_wrap=True)
            
            for case_id in active_cases:
                table.add_row(case_id)
            
            console.print(table)
        
        await agent.cleanup()
    
    asyncio.run(_list())


@app.command()
def close_case(
    case_id: str = typer.Argument(..., help="Case identifier"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Close a case."""
    async def _close():
        agent = create_agent(config_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing agent...", total=None)
            await agent.initialize()
            
            progress.add_task("Closing case...", total=None)
            result = await agent.close_case(validate_case_id(case_id))
        
        console.print(Panel(
            f"[green]Case closed successfully![/green]\n\n"
            f"Case ID: {result['case_id']}\n"
            f"Status: {result['status']}\n"
            f"Final report generated and saved.",
            title="Case Closed"
        ))
        
        await agent.cleanup()
    
    asyncio.run(_close())


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()